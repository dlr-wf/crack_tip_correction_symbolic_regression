import json
import sys
from datetime import datetime
import os
import subprocess
import numpy as np
import getpass


class AnsysSimulation:
    """Class for Simulations in ANSYS MECHANICAL.

    Methods:
        * export_nodemap - saves Ansys node data into txt-file 'Nodemap'
        * plot - plots mesh, displacements, and von-Mises strain
        * solve - solution of PDE

    """

    def __init__(self, mapdl, OUTPUT_PATH):
        """Class for Simulations in ANSYS MECHANICAL.

        Args:
            mapdl: PyAnsys MAPDL instance

        """
        self.total_number_elements = None
        self.total_number_nodes = None
        self.ct_csys = None
        self.keyopt_3 = None
        self.element_type = None
        self.end_time = None
        self.start_time = datetime.now()
        self.cint_post = {}
        self.ct_num = 0
        self.ct_center_csys = None
        self.cint_crack_tips = []
        self.cint_id = 0
        self.OUTPUT_PATH = OUTPUT_PATH
        self.sigma_xx = None
        self.sigma_yy = None
        self.sigma_xy = None
        self.esize = None
        self.top_edge = None
        self.bottom_edge = None
        self.right_edge = None
        self.left_edge = None
        self.width = None
        self.height = None
        self.a_w_ratio = None
        self.thickness = 1.0  # mm
        self.youngs_modulus = None
        self.nuxy = None

        # Initialization
        self.mapdl = mapdl
        self.mapdl.finish()
        self.mapdl.clear()
        self.mapdl.prep7()
        self.mapdl.seltol(0.01)

    def exit(self):
        self.mapdl.exit()

    def set_material_elements(self, youngs_modulus=73000, nuxy=0.33):
        """Sets variables and parameters for Ansys run.

        Args:
            youngs_modulus: Young's modulus of the material (N/mm^2)
            nuxy: Poisson ratio of the material

        """

        # Material properties
        self.youngs_modulus = youngs_modulus
        self.nuxy = nuxy

        self.mapdl.mp("EX", 1, self.youngs_modulus)  # Young modulus (N/mm^2)
        self.mapdl.mp("NUXY", 1, self.nuxy)  # Poisson ration

        # Element types
        self.element_type = "PLANE182"  # 2-D 4-Node Structural Solid
        self.mapdl.et(1, self.element_type)
        self.keyopt_3 = 3  # plane stress with thickness input
        self.mapdl.keyopt(1, 3, self.keyopt_3)
        self.mapdl.r(1, self.thickness)

    def set_geometry(self, width=50, height=50, a_w_ratio=0.5):
        """Sets key points for Ansys run. Crack tip is at (0,0) in global coordinate system.

        Args:
            width: width of the model (mm)
            height: height of the model (mm)
            a_w_ratio: ratio of crack length to width
        """
        self.width = width
        self.height = height
        self.a_w_ratio = a_w_ratio

        # compute edge coordinates
        self.left_edge = -self.width * a_w_ratio
        self.right_edge = width * (1.0 - a_w_ratio)
        self.bottom_edge = -height / 2.0
        self.top_edge = height / 2.0

        # set coordinate system
        self.mapdl.csys(0)

        # the crack tip is always in the centre of the global coordinate system
        self.ct_csys = 0

        # set key points
        KP_1 = self.mapdl.k("", 0.0, 0.0)
        KP_2 = self.mapdl.k("", self.left_edge, 0.0)
        KP_3 = self.mapdl.k("", self.left_edge, 0.0)
        KP_4 = self.mapdl.k("", self.left_edge, self.top_edge)
        KP_5 = self.mapdl.k("", 0.0, self.top_edge)
        KP_6 = self.mapdl.k("", self.right_edge, self.top_edge)
        KP_7 = self.mapdl.k("", self.right_edge, 0.0)
        KP_8 = self.mapdl.k("", self.right_edge, self.bottom_edge)
        KP_9 = self.mapdl.k("", 0.0, self.bottom_edge)
        KP_10 = self.mapdl.k("", self.left_edge, self.bottom_edge)

        # set areas
        self.mapdl.a(KP_1, KP_2, KP_4, KP_5)
        self.mapdl.a(KP_1, KP_5, KP_6, KP_7)
        self.mapdl.a(KP_1, KP_7, KP_8, KP_9)
        self.mapdl.a(KP_1, KP_9, KP_10, KP_3)

    def set_mesh(self, esize=1.0):
        """Sets meshing parameters for Ansys run.

        Args:
            esize: element size (mm)

        """
        # meshing
        self.esize = esize
        self.mapdl.type(1)
        self.mapdl.mat(1)
        self.mapdl.r(1)
        self.mapdl.allsel()
        self.mapdl.esize(self.esize)
        self.mapdl.amesh("ALL")
        self.total_number_nodes = self.mapdl.get(entity="NODE", entnum="", item1="COUNT")
        self.total_number_elements = self.mapdl.get(entity="ELEM", entnum="", item1="COUNT")

    def set_boundary_conditions(self, sigma_xx=0.0, sigma_yy=0.0, sigma_xy=0.0):
        """Sets boundary conditions for Ansys run.

        Args:
            sigma_xx: normal stress in x-direction (N/mm^2)
            sigma_yy: normal stress in y-direction (N/mm^2)
            sigma_xy: shear stress in xy-direction (N/mm^2)

        """

        self.sigma_xx = sigma_xx
        self.sigma_yy = sigma_yy
        self.sigma_xy = sigma_xy

        # boundary conditions
        self.mapdl.csys(0)
        self.mapdl.seltol(0.001)

        # fix crack tip
        self.mapdl.nsel("S", "LOC", "X", 0.0)
        self.mapdl.nsel("R", "LOC", "Y", 0.0)
        self.mapdl.d("ALL", "UX")
        self.mapdl.d("ALL", "UY")

        # fix right edge
        self.mapdl.nsel("S", "LOC", "X", self.right_edge)
        self.mapdl.nsel("R", "LOC", "Y", 0.0)
        self.mapdl.d("ALL", "UY")

        # top
        self.mapdl.nsel("S", "LOC", "Y", self.top_edge)
        number_nodes = self.mapdl.get(entity='node', item1='count')
        element_area = self.width * self.thickness / number_nodes
        self.mapdl.f("ALL", "FX", element_area * self.sigma_xy)
        self.mapdl.f("ALL", "FY", element_area * self.sigma_yy)

        # bottom
        self.mapdl.nsel("S", "LOC", "Y", self.bottom_edge)
        number_nodes = self.mapdl.get(entity='node', item1='count')
        element_area = self.width * self.thickness / number_nodes
        self.mapdl.f("ALL", "FX", -element_area * self.sigma_xy)
        self.mapdl.f("ALL", "FY", -element_area * self.sigma_yy)

        # right
        self.mapdl.nsel("S", "LOC", "X", self.right_edge)
        number_nodes = self.mapdl.get(entity='node', item1='count')
        element_area = self.height * self.thickness / number_nodes
        self.mapdl.f("ALL", "FX", element_area * self.sigma_xx)
        self.mapdl.f("ALL", "FY", element_area * self.sigma_xy)

        # left
        self.mapdl.nsel("S", "LOC", "X", self.left_edge)
        number_nodes = self.mapdl.get(entity='node', item1='count')
        element_area = self.height * self.thickness / number_nodes
        self.mapdl.f("ALL", "FX", -element_area * self.sigma_xx)
        self.mapdl.f("ALL", "FY", -element_area * self.sigma_xy)
        self.mapdl.allsel()

        # set fracture mechanics computation
        self.cint_crack_tips.append(self.set_fracture_mechanics(name='crack_tip_1', clocal=self.ct_csys))

    def set_fracture_mechanics(self, name: str, clocal: int):
        """Sets fracture mechanics parameters for Ansys run.

        Args:
            name: name of the fracture mechanics calculation
            clocal: coordinate system of the crack tip

        """
        # Crack tip component
        self.mapdl.csys(clocal)
        self.ct_num = self.ct_num + 1
        crack_tip_component = f"CT_CM_{int(self.ct_num)}"
        self.mapdl.nsel("S", "LOC", "X", 0)
        nodes = self.mapdl.nsel("R", "LOC", "Y", 0)
        self.mapdl.cm(crack_tip_component, "node")

        # Crack tip assist node
        self.mapdl.csys(0)
        self.mapdl.nsel("S", "LOC", "X", self.left_edge)
        self.mapdl.nsel("R", "LOC", "Y", 0)
        assist_node = self.mapdl.get("AssistNode", "NODE", 0, "NUM", "max")

        self.cint_id = self.cint_id + 1
        cint_id_VCCT = self.cint_id
        self.cint_id = self.cint_id + 1
        cint_id_JINT = self.cint_id
        self.cint_id = self.cint_id + 1
        cint_id_SIFS = self.cint_id
        self.cint_id = self.cint_id + 1
        cint_id_TSTRESS = self.cint_id

        # Define a new VCCT calculation
        self.mapdl.cint("NEW", cint_id_VCCT)
        self.mapdl.cint("TYPE", "VCCT")
        self.mapdl.cint("CTNC", crack_tip_component, assist_node, 1)
        self.mapdl.cint("NORM", clocal, 2)
        self.mapdl.cint("SYMM", "OFF")

        # Define a new JINT calculation
        self.mapdl.cint("NEW", cint_id_JINT)
        self.mapdl.cint("TYPE", "JINT")
        self.mapdl.cint("CTNC", crack_tip_component, assist_node, 1)
        self.mapdl.cint("NORM", clocal, 2)
        self.mapdl.cint("NCON", 6)
        self.mapdl.cint("SYMM", "OFF")

        # Define a new JINT calculation
        self.mapdl.cint("NEW", cint_id_SIFS)
        self.mapdl.cint("TYPE", "SIFS")
        self.mapdl.cint("CTNC", crack_tip_component, assist_node, 1)
        self.mapdl.cint("NORM", clocal, 2)
        self.mapdl.cint("NCON", 6)
        self.mapdl.cint("SYMM", "OFF")

        # Define a new T-Stress calculation
        self.mapdl.cint("NEW", cint_id_TSTRESS)
        self.mapdl.cint("TYPE", "TSTRESS")
        self.mapdl.cint("CTNC", crack_tip_component, assist_node, 1)
        self.mapdl.cint("NORM", clocal, 2)
        self.mapdl.cint("NCON", 6)
        self.mapdl.cint("SYMM", "OFF")
        self.mapdl.csys(0)

        cint_params = {"name": name,
                       "crack_tip_component": crack_tip_component,
                       "assist_node": assist_node,
                       "nodes": nodes,
                       "VCCT_ID": cint_id_VCCT,
                       "JINT_ID": cint_id_JINT,
                       "SIFS_ID": cint_id_SIFS,
                       "TSTRESS_ID": cint_id_TSTRESS}

        return cint_params

    def solve(self):
        self.mapdl.slashsolu()
        self.mapdl.antype("STATIC")
        self.mapdl.allsel()
        self.mapdl.outres('All', 'All')
        self.mapdl.solve()

    def export_nodemap(self, filename) -> None:
        """Saves the current Ansys node data into a txt-file called 'Nodemap'.

        Args:
            "filename" (str): Name of the nodemap file.
        """

        # Change to post-processing
        self.mapdl.post1()
        self.mapdl.set('last')
        self.mapdl.allsel()

        # Get coordinates, displacements, strains, and stresses
        node_list = self.mapdl.get_array(entity='NODE', item1='NLIST')
        node_loc_x = self.mapdl.get_array(entity='NODE', item1='LOC', it1num='X')
        node_loc_y = self.mapdl.get_array(entity='NODE', item1='LOC', it1num='Y')
        node_loc_z = self.mapdl.get_array(entity='NODE', item1='LOC', it1num='Z')

        u_x = self.mapdl.get_array(entity='NODE', item1='U', it1num='X')
        u_y = self.mapdl.get_array(entity='NODE', item1='U', it1num='Y')
        u_z = self.mapdl.get_array(entity='NODE', item1='U', it1num='Z')

        epto_xx = self.mapdl.get_array(entity='NODE', item1='EPTO', it1num='X') * 100.0
        epto_yy = self.mapdl.get_array(entity='NODE', item1='EPTO', it1num='Y') * 100.0
        epto_xy = self.mapdl.get_array(entity='NODE', item1='EPTO', it1num='XY') * 0.5
        epto_eqv = self.mapdl.get_array(entity='NODE', item1='EPTO', it1num='EQV') * 100.0

        s_xx = self.mapdl.get_array(entity='NODE', item1='S', it1num='X')
        s_yy = self.mapdl.get_array(entity='NODE', item1='S', it1num='Y')
        s_xy = self.mapdl.get_array(entity='NODE', item1='S', it1num='XY')
        s_eqv = self.mapdl.get_array(entity='NODE', item1='S', it1num='EQV')

        export_data = np.asarray([node_list, node_loc_x, node_loc_y, node_loc_z, u_x, u_y, u_z,
                                  epto_xx, epto_yy, epto_xy, epto_eqv, s_xx, s_yy, s_xy, s_eqv]).T

        with open(os.path.join(self.OUTPUT_PATH, f'{filename}_nodemap.txt'), 'w') as file:
            file.write('# Header for nodemap file\n')
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            file.write(f'# {"date":>16}: {current_time}\n')
            file.write(f'# {"width":>16}: {self.width:15.4f}\n')
            file.write(f'# {"height":>16}: {self.height:15.4f}\n')
            file.write(f'# {"thickness":>16}: {self.thickness:15.4f}\n')
            file.write(f'# {"a_w_ratio":>16}: {self.a_w_ratio:15.4f}\n')
            file.write(f'# {"esize":>16}: {self.esize:15.4f}\n')
            file.write(f'# {"youngs_modulus":>16}: {self.youngs_modulus:15.4f}\n')
            file.write(f'# {"poisson_ratio":>16}: {self.nuxy:15.4f}\n')
            file.write(f'# {"sigma_xx":>16}: {self.sigma_xx:15.4f}\n')
            file.write(f'# {"sigma_yy":>16}: {self.sigma_yy:15.4f}\n')
            file.write(f'# {"sigma_xy":>16}: {self.sigma_xy:15.4f}\n')
            file.write(f'# \n')
            file.write(f'# {"index":>10}; '
                       f'{"x_undef":>12s}; {"y_undef":>12s}; {"z_undef":>12s}; '
                       f'{"ux":>12s}; {"uy":>12s}; {"uz":>12s}; '
                       f'{"eps_x":>12s}; {"eps_y":>12s}; {"eps_xy":>12s}; {"eps_eqv":>12s}; '
                       f'{"s_x":>12s}; {"s_y":>12s}; {"s_xy":>12s}; {"s_eqv":>12s}\n')

            for i, line in enumerate(export_data):
                file.write(f'{line[0]:12.1f}; {line[1]:12.6f}; {line[2]:12.6f}; {line[3]:12.6f}; '
                           f'{line[4]:12.6f}; {line[5]:12.6f}; {line[6]:12.6f}; '
                           f'{line[7]:12.6f}; {line[8]:12.6f}; {line[9]:12.6f}; {line[10]:12.6f}; '
                           f'{line[11]:12.6f}; {line[12]:12.6f}; {line[13]:12.6f}; {line[14]:12.6f}\n')

    def export_vtk(self, filename) -> None:
        """Saves the current Ansys results as vtk file.

        Args:
            "filename" (str): Name of the vtk file.
        """

        self.mapdl.post1()
        self.mapdl.set("LAST")
        self.mapdl.allsel()

        # Define PyVista unstructured grid object and fill it with data
        grid = self.mapdl.mesh.grid
        grid.point_data['node_list'] = self.mapdl.get_array(entity='NODE', item1='NLIST')
        grid.point_data['u_x'] = self.mapdl.get_array(entity='NODE', item1='U', it1num='X')
        grid.point_data['u_y'] = self.mapdl.get_array(entity='NODE', item1='U', it1num='Y')
        grid.point_data['u_z'] = self.mapdl.get_array(entity='NODE', item1='U', it1num='Z')
        grid.point_data['epto_xx'] = self.mapdl.get_array(entity='NODE', item1='EPTO', it1num='X') * 100.0
        grid.point_data['epto_yy'] = self.mapdl.get_array(entity='NODE', item1='EPTO', it1num='Y') * 100.0
        grid.point_data['epto_xy'] = self.mapdl.get_array(entity='NODE', item1='EPTO', it1num='XY') * 0.5
        grid.point_data['epto_eqv'] = self.mapdl.get_array(entity='NODE', item1='EPTO', it1num='EQV') * 100.0
        grid.point_data['s_xx'] = self.mapdl.get_array(entity='NODE', item1='S', it1num='X')
        grid.point_data['s_yy'] = self.mapdl.get_array(entity='NODE', item1='S', it1num='Y')
        grid.point_data['s_xy'] = self.mapdl.get_array(entity='NODE', item1='S', it1num='XY')
        grid.point_data['s_eqv'] = self.mapdl.get_array(entity='NODE', item1='S', it1num='EQV')
        grid.save(os.path.join(self.OUTPUT_PATH, f'{filename}_mesh.vtk'), binary=False)
        return grid

    def postprocessing_cint(self):

        self.mapdl.post1()
        self.mapdl.set('last')
        self.mapdl.csys(0)
        self.mapdl.rsys()

        for cint_crack_tip in self.cint_crack_tips:
            crack_tip_component = cint_crack_tip['crack_tip_component']
            cint_ent = cint_crack_tip['name']
            self.cint_post[cint_ent] = {}
            self.mapdl.cmsel('S', crack_tip_component, 'NODE')
            crack_tip_nodes = self.mapdl.nlist('ALL').to_array()[:, 0]
            for crack_tip_node in crack_tip_nodes:
                crack_tip_node = int(crack_tip_node)
                self.cint_post[cint_ent][crack_tip_node] = {}
                self.cint_post[cint_ent][crack_tip_node]['node'] = crack_tip_node
                self.cint_post[cint_ent][crack_tip_node]['x'] = self.mapdl.queries.nx(crack_tip_node)
                self.cint_post[cint_ent][crack_tip_node]['y'] = self.mapdl.queries.ny(crack_tip_node)
                self.cint_post[cint_ent][crack_tip_node]['z'] = self.mapdl.queries.nz(crack_tip_node)

                G_1_VCCT = self.mapdl.get(entity='CINT', entnum=cint_crack_tip['VCCT_ID'],
                                          it1num=crack_tip_node, it2num=1, it3num='G1')
                G_2_VCCT = self.mapdl.get(entity='CINT', entnum=cint_crack_tip['VCCT_ID'],
                                          it1num=crack_tip_node, it2num=1, it3num='G2')
                G_3_VCCT = self.mapdl.get(entity='CINT', entnum=cint_crack_tip['VCCT_ID'],
                                          it1num=crack_tip_node, it2num=1, it3num='G3')

                K_I_VCCT = np.sqrt(np.abs(G_1_VCCT) * self.youngs_modulus / 1000)
                K_II_VCCT = np.sqrt(np.abs(G_2_VCCT) * self.youngs_modulus / 1000)
                K_III_VCCT = np.sqrt(np.abs(G_3_VCCT) * self.youngs_modulus / 1000)

                J = []
                K_1 = []
                K_2 = []
                K_3 = []
                TSTRESS = []
                for contour in range(1, 7):
                    J.append(self.mapdl.get(entity='CINT', entnum=cint_crack_tip['JINT_ID'], it1num=crack_tip_node,
                                            it2num=contour, it3num='JINT'))
                    K_1.append(self.mapdl.get(entity='CINT', entnum=cint_crack_tip['SIFS_ID'], it1num=crack_tip_node,
                                              it2num=contour, it3num='K1'))
                    K_2.append(self.mapdl.get(entity='CINT', entnum=cint_crack_tip['SIFS_ID'], it1num=crack_tip_node,
                                              it2num=contour, it3num='K2'))
                    K_3.append(self.mapdl.get(entity='CINT', entnum=cint_crack_tip['SIFS_ID'], it1num=crack_tip_node,
                                              it2num=contour, it3num='K3'))
                    TSTRESS.append(
                        self.mapdl.get(entity='CINT', entnum=cint_crack_tip['TSTRESS_ID'], it1num=crack_tip_node,
                                       it2num=contour, it3num='TSTRESS'))

                K_I_SIFS = np.mean(K_1[-4:-1]) / np.sqrt(1000)
                K_II_SIFS = np.mean(K_2[-4:-1]) / np.sqrt(1000)
                K_III_SIFS = np.mean(K_3[-4:-1]) / np.sqrt(1000)
                T_MEAN = np.mean(TSTRESS[-4:-1])
                K_V_VCCT = K_I_VCCT / 2 + 0.5 * np.sqrt(
                    np.abs(K_I_VCCT ** 2 + 5.336 * K_II_VCCT ** 2 + 4 * K_III_VCCT ** 2))
                K_V_SIFS = K_I_SIFS / 2 + 0.5 * np.sqrt(
                    np.abs(K_I_SIFS ** 2 + 5.336 * K_II_SIFS ** 2 + 4 * K_III_SIFS ** 2))

                self.cint_post[cint_ent][crack_tip_node]['G_1_VCCT'] = G_1_VCCT
                self.cint_post[cint_ent][crack_tip_node]['G_2_VCCT'] = G_2_VCCT
                self.cint_post[cint_ent][crack_tip_node]['G_3_VCCT'] = G_3_VCCT
                self.cint_post[cint_ent][crack_tip_node][f'J_contours'] = J
                self.cint_post[cint_ent][crack_tip_node][f'K_1_contours'] = K_1
                self.cint_post[cint_ent][crack_tip_node][f'K_2_contours'] = K_2
                self.cint_post[cint_ent][crack_tip_node][f'K_3_contours'] = K_3
                self.cint_post[cint_ent][crack_tip_node][f'TSTRESS_contours'] = TSTRESS
                self.cint_post[cint_ent][crack_tip_node][f'K_I_VCCT'] = K_I_VCCT
                self.cint_post[cint_ent][crack_tip_node][f'K_II_VCCT'] = K_II_VCCT
                self.cint_post[cint_ent][crack_tip_node][f'K_III_VCCT'] = K_III_VCCT
                self.cint_post[cint_ent][crack_tip_node][f'K_I_SIFS'] = K_I_SIFS
                self.cint_post[cint_ent][crack_tip_node][f'K_II_SIFS'] = K_II_SIFS
                self.cint_post[cint_ent][crack_tip_node][f'K_III_SIFS'] = K_III_SIFS
                self.cint_post[cint_ent][crack_tip_node][f'T_MEAN'] = T_MEAN
                self.cint_post[cint_ent][crack_tip_node][f'K_V_VCCT'] = K_V_VCCT
                self.cint_post[cint_ent][crack_tip_node][f'K_V_SIFS'] = K_V_SIFS

    def plot(self, max_stress: float, filename: str = 'out') -> None:
        """Plots the mesh, displacements, and von-Mises strain.

        Args:
            max_stress: of the material (used as upper limit for stress plot)
            filename: name of the output file

        """

        self.mapdl.post1()
        self.mapdl.set('LAST')
        self.mapdl.allsel()
        self.mapdl.esel('S', 'TYPE', vmin=1)  # mask out holes

        # plot element mesh
        self.mapdl.eplot(cpos='xy', savefig=os.path.join(self.OUTPUT_PATH, f'{filename}_mesh.png'))

        # colorbar settings
        sbar_kwargs = dict(
            title_font_size=20,
            label_font_size=16,
            n_labels=5,
            font_family="arial",
            color="black"
        )

        # plot nodal von-Mises stress
        plotter = self.mapdl.post_processing.plot_nodal_eqv_stress(
            cpos='xy',
            background='white',
            scalar_bar_args=sbar_kwargs,
            show_axes=True,
            n_colors=256,
            off_screen=True,
            cmap="coolwarm",
            return_plotter=True,
        )
        plotter.update_scalar_bar_range([0, max_stress])
        plotter.screenshot(os.path.join(self.OUTPUT_PATH, f'{filename}_stress_vm.png'))

        # plot nodal y-displacement
        self.mapdl.post_processing.plot_nodal_displacement(
            component='Y',
            cpos='xy',
            background='white',
            scalar_bar_args=sbar_kwargs,
            show_axes=True,
            n_colors=256,
            off_screen=True,
            cmap="coolwarm",
            savefig=os.path.join(self.OUTPUT_PATH, f'{filename}_y_displacement.png')
        )
        # plot nodal x-displacement
        self.mapdl.post_processing.plot_nodal_displacement(
            component='X',
            cpos='xy',
            background='white',
            scalar_bar_args=sbar_kwargs,
            show_axes=True,
            n_colors=256,
            off_screen=True,
            cmap="coolwarm",
            savefig=os.path.join(self.OUTPUT_PATH, f'{filename}_x_displacement.png')
        )

    def _convert_to_serializable(self, data):
        """Converts data to a serializable format.

        Args:
            data: data to be converted

        Returns:
            serializable_data: converted data

        """

        if isinstance(data, (list, np.ndarray)):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._convert_to_serializable(value) for key, value in data.items()}
        elif isinstance(data, (float, np.float64)):
            return float(data)
        elif isinstance(data, (int, np.integer)):
            return int(data)
        else:
            return str(data)

    def export_results(self, filename: str):
        """Exports the results of the Ansys run into a json-file.

        Args:
            filename: name of the output file

        """

        self.end_time = datetime.now()

        serializable_data = {}
        for attr, value in vars(self).items():
            if not callable(value) and not attr.startswith('__'):
                serializable_data[attr] = self._convert_to_serializable(value)

        with open(os.path.join(self.OUTPUT_PATH, f'{filename}_summary.json'), 'w') as json_file:
            json.dump(serializable_data, json_file, indent=4)


def kill_ansys():
    """Kill all running ANSYS processes for the current user."""
    print('Killing all ANSYS processes for current user...')

    # exclusion list
    excluded_str = []
    # identifier for python main process
    main_script_str = os.path.join(os.getcwd(), os.path.basename(sys.argv[0]))
    excluded_str.append(main_script_str)

    p = subprocess.Popen(['ps', '-eo', 'pid,user,command'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.decode().splitlines():
        # exclud exceptions
        if any([excluded in line for excluded in excluded_str]):
            continue
        if 'ansys' in line.lower():
            pid, user, cmd = line.split(None, 2)
            if user == getpass.getuser():
                try:
                    os.kill(int(pid), 9)
                except ProcessLookupError:
                    pass
                print(f'Killed process {pid} for user {user} with command {cmd}')


def delete_ansys_leftovers(ansys_folder: str):
    """Delete overhead Ansys files after the run.

    Args:
        ansys_folder: path of ansys run location

    """

    files = os.listdir(ansys_folder)
    keep_endings = ('.txt', '.png', '.vtk', '.pdf', '.eps', '.svg')
    for ansys_file in files:
        if not ansys_file.endswith(keep_endings):
            os.remove(os.path.join(ansys_folder, ansys_file))


def delete_folder(folder: str):
    """Delete folder and all files in it.

    Args:
        folder: path of folder to delete

    """
    assert os.path.isdir(folder), f'{folder} is not a folder'
    files = os.listdir(folder)
    for file in files:
        os.remove(os.path.join(folder, file))
    os.rmdir(folder)
