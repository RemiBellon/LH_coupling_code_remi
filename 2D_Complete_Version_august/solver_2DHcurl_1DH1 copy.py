
'''
Solver_2DHcurl_1DH1 is a class that gather the functions to:
    - Build the mesh (cf. build_mesh_with_PMLs for more infos) based on 2D box sizes (plasma & pmls) in config_dict.py (cfg)
    - Build the physics (cf. build_physics) based on B field, ne, omega_LH (specified in config_dict.py) and Stix cold plasma approx in generalized cartesian coordinates.
    - Initialize and solve the Helmholtz wave equation on cartesian mesh (cf. solve_helmholtz_2DHcurl_1DH1_with_pml)
'''

from matplotlib.pyplot import step
import netgen.occ as occ
from ngsolve import *
import numpy as np

class LHCouplingSolver_2DHcurl_1DH1:
    def __init__(self, config_dict, geom_mode, box_medium, antenna_grill):
        self.cfg = config_dict          # type = dict ==> dictionnary of dictionnaries with physics (wave, plasma) and geometry (domain, pmls) values
        self.geom_mode = geom_mode          # 1D or 2D
        self.box_medium = box_medium        # "VACUUM", "PLASMA"
        self.antenna_grill = antenna_grill

        self.mesh = None                # type = ngsolve.comp.Mesh ==> Mesh to solve wave equation
        self.fes = None                 # type = ngsolve.comp.FESpace ==> Hcurl space function to solve wave equation
        self.E_field = None             # type = ngsolve.comp.GridFunction ==> Solution of the wave equation on the mesh/grid

        self.wg_medium = self.cfg['DOMAIN'].get('wg_medium', 'VACUUM').upper()
        self.x = x                      # type = ngsolve.fem.CoefficientFunction ==> Space coords to compute wave equation variables
        self.z = y                      # In the context y is out of 2D plane direction. In 2D only the plane (xOz) is describe.

        self.n_para, self.n_perp_p, self.n_perp_m = self.compute_physics_parameters()
        # print(f'In LHCoupling class: n_para: {self.n_para:.1f}, n_perp_p: {self.n_perp_p:.2e}, n_perp_m: {self.n_perp_m:.2e}')
        if geom_mode != "1D" and antenna_grill is not None:
            print(f'Waveguide Medium set to: {self.wg_medium}')

        # Set the waveguide length if antenna or not
        if self.antenna_grill is None:
            self.Lx_wg = 0.0            # No waveguide(s)
        else:
            self.Lx_wg = self.cfg['DOMAIN'].get('Lx_wg', 0.03)

    def compute_physics_parameters(self) -> None:
        self.omega_LH, self.k0, self.B0 = self.cfg['WAVE']['omega_LH'], self.cfg['WAVE']['k0'], self.cfg['PLASMA']['B0']
        self.qe, self.me, self.mi, self.eps0, self.mu0 = self.cfg['CONST']['qe'], self.cfg['CONST']['me'], self.cfg['CONST']['mi'], self.cfg['CONST']['eps0'], self.cfg['CONST']['mu0']

        # Récupération des densités limites (Bord et Cœur) depuis la configuration
        if self.box_medium == "VACUUM":
            self.ne_edge = 0.0
            self.ne_core = 0.0
        else:
            self.ne_edge = self.cfg['PLASMA']['ne_points'][0][1]
            self.ne_core = self.cfg['PLASMA']['ne_points'][-1][1]

        self.Om_ce = (self.qe * self.B0) / self.me
        self.Om_ci = (self.qe * self.B0) / self.mi

        # On extrait n_para ici pour pouvoir l'utiliser dans la fonction interne
        self.n_para = self.cfg['WAVE']['n_para']

        # --- FONCTION INTERNE POUR CALCULER LES DEUX RACINES DE BOOKER ---
        def get_n_perp_at_density(ne_val):
            n_para_sq = self.n_para**2

            if ne_val == 0:
                # Dans le vide, les deux modes dégénèrent sur la même solution d'onde plane
                n_perp_vac = np.sqrt(complex(1.0 - n_para_sq))
                return 1.0, 0.0, 1.0, n_perp_vac, n_perp_vac

            w_pe2 = (ne_val * self.qe**2) / (self.me * self.eps0)
            w_pi2 = (ne_val * self.qe**2) / (self.mi * self.eps0)

            D = -(self.Om_ce * w_pe2)/(self.omega_LH*(self.omega_LH**2 - self.Om_ce**2)) + \
                 (self.Om_ci * w_pi2)/(self.omega_LH*(self.omega_LH**2 - self.Om_ci**2))
            S = 1 - w_pe2/(self.omega_LH**2 - self.Om_ce**2) - w_pi2/(self.omega_LH**2 - self.Om_ci**2)
            P = 1 - w_pe2/self.omega_LH**2 - w_pi2/self.omega_LH**2

            # Équation de Booker exacte (Correction du "2.0**2" codé en dur)
            B_stix = (S + P)*n_para_sq - (S**2 - D**2) - P*S
            C_stix = P * (n_para_sq - (S + D)) * (n_para_sq - (S - D))

            delta = B_stix**2 - 4*S*C_stix

            # Onde lente (Slow Wave) -> Signe "+" devant la racine
            n_perp_p_val = np.sqrt(complex((-B_stix + np.sqrt(complex(delta))) / (2*S)))

            # Onde rapide (Fast Wave) -> Signe "-" devant la racine
            n_perp_m_val = np.sqrt(complex((-B_stix - np.sqrt(complex(delta))) / (2*S)))

            return S, D, P, n_perp_p_val, n_perp_m_val

        # 1. Propriétés au BORD (pour l'admittance de l'antenne et les conditions aux limites)
        self.S_edge, self.D_edge, self.P_edge, self.n_perp_p_edge, self.n_perp_m_edge = get_n_perp_at_density(self.ne_edge)

        # On remplace les variables globales par les valeurs au bord
        self.S, self.D, self.P = self.S_edge, self.D_edge, self.P_edge
        self.n_perp_p = self.n_perp_p_edge
        self.n_perp_m = self.n_perp_m_edge

        # 2. Propriétés au CŒUR (pour dimensionner la PML radiale et le maillage fin)
        _, _, _, self.n_perp_p_core, self.n_perp_m_core = get_n_perp_at_density(self.ne_core)

        return self.n_para, self.n_perp_p, self.n_perp_m

    def plot_radial_density_profile(self, num_points=1000) -> None:
        '''
        Génère une coupe radiale 1D (le long de l'axe x) pour visualiser le profil
        de densité extrait directement de la CoefficientFunction maillée.
        '''
        import matplotlib.pyplot as plt

        if self.mesh is None:
            print("[!] Erreur : Le maillage doit être généré (.build_mesh_with_PMLs()) avant de pouvoir tracer le profil.")
            return

        # Évaluation au milieu de la boîte en Z (axe toroidal)
        z_eval = self.Lz_plasma_src * 0.5 + self.Lz_wall

        # Plage d'échantillonnage de l'arrière des guides (si existants) jusqu'à la fin de la PML radiale
        x_coords = np.linspace(-self.Lx_wg, self.Lx_tot, num_points)
        ne_values = []

        for xi in x_coords:
            try:
                mip = self.mesh(xi, z_eval)
                ne_values.append(self.ne_profile_cf(mip))
            except Exception:
                ne_values.append(0.0) # Zone hors maillage / parois métalliques

        # Tracé du graphique
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(x_coords, ne_values, lw=2.5, color='darkblue', label='Bi-linear radiale density profile $n_e(x)$')
        #ax.set_yscale('log')

        # Lignes repères physiques
        ax.axvline(x=0.0, color='black', linestyle=':', lw=1.5)
        ax.text(0.0, max(ne_values)*0.35, 'Antenna/Plasma interface', rotation=90, va='bottom', fontsize=14)

        ax.axvline(x=self.Lx_plasma, color='crimson', linestyle='--', lw=1.5)
        ax.text(self.Lx_plasma, max(ne_values)*0.35, 'Plasma/PML interface', rotation=90, va='bottom', color='crimson', fontsize=14)
        ax.set_ylim(0, max(ne_values)*1.1)
        # Repères des nœuds de gradients injectés
        for idx, pt in enumerate(self.cfg['PLASMA']['ne_points']):
            ax.plot(pt[0], pt[1], marker='o', color='gold', markersize=8, markeredgecolor='black', zorder=5)
            ax.text(pt[0], pt[2] if len(pt)>2 else pt[1], f'', ha='left', va='bottom', fontsize=9)

        ax.set_xlabel("Radial Position $x$ [m]", fontsize=16)
        ax.set_ylabel("Plasma Density $n_e$ [$m^{-3}$]", fontsize=16)
        # ax.set_title("Coupe Radiale du Profil de Densité Plasma (Vérification NGSolve)", fontsize=13, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='best')
        plt.savefig("radial_density_profile.pdf", dpi=300)
        plt.tight_layout()
        plt.show()

    # =====================================================================
    # MESH GENERATION (Plasma + PML Domains)
    # =====================================================================
    def build_mesh_with_PMLs(self) -> None:
        '''
            Create the mesh object for C++ NGSolve solver:
                - Recover Domain sizes from cfg_dict.py
                - Discretize the 2D Domain in Plasma & PMLs areas
                - Define the external edges to set up boundary conditions
                - Glue all areas and set the mesh resolution based on the smallest wavelength
        '''
        # print(f'geom_mode: {self.geom_mode}')
        # print(f'box_medium: {self.box_medium}')

        self.lambda0 = self.cfg['WAVE']['lambda0']
        self.lambda_para = self.lambda0 / max(abs(self.n_para), 1e-6)
        self.lambda_perp_m = self.lambda0 / np.abs(self.n_perp_m)

        # --- NOUVEAUTÉ : Extraction du paramètre d'évaluation (Coeur) ---
        # On cherche la plus petite longueur d'onde pour dimensionner la PML et le maillage.
        # S'il y a un gradient, c'est le coeur (n_perp_p_core) qui dicte la physique limite.
        n_perp_eval = getattr(self, 'n_perp_p_core', self.n_perp_p)
        self.lambda_perp_eval = self.lambda0 / np.abs(n_perp_eval)

        self.Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
        if self.box_medium == 'PLASMA':
            # print('[!] MEDIUM = PLASMA, Lx_pml is set based on core lambda_perp')
            # La PML radiale absorbe l'onde venant du coeur, elle doit être dimensionnée par rapport au coeur.
            self.Lx_pml = self.cfg['DOMAIN']['Lx_pml'] = 1 * self.lambda_perp_eval
        elif self.box_medium == 'VACUUM':
            print('[!] MEDIUM = VACUUM, Lx_pml is set based on freq_LH')
            self.Lx_pml = self.cfg['DOMAIN']['Lx_pml'] = 1 * self.cfg['CONST']['c0'] / self.cfg['WAVE']['freq_LH']

        self.cfg['DOMAIN']['Lx_tot'] = self.Lx_tot = self.Lx_plasma + self.Lx_pml

        if self.box_medium == "VACUUM":
            if self.geom_mode == "1D":
                self.Lz_plasma_src = self.cfg['DOMAIN']['Lz_plasma']
                self.Lz_pml = self.cfg['DOMAIN']['Lz_pml'] = 0.
                self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = 0.
            elif self.geom_mode == "2D":
                self.Lz_plasma_src = self.cfg['DOMAIN']['Lz_plasma']
                self.Lz_pml = self.cfg['DOMAIN']['Lz_pml'] = self.Lz_plasma_src / 2
                self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = 0.

        elif self.box_medium == "PLASMA":
            if self.geom_mode == "1D":
                self.Lz_plasma_src = self.cfg['DOMAIN']['Lz_plasma'] = 3. * self.lambda_para
                self.Lz_pml = self.cfg['DOMAIN']['Lz_pml'] = 0.
                self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = 0.

            elif self.geom_mode == "2D":
                self.Lz_pml = self.cfg['DOMAIN']['Lz_pml']

                if self.antenna_grill is not None:
                    base_instruction = self.antenna_grill.generate_mesh_instructions(z_start_position=0.0)
                    self.cfg['DOMAIN']['Lz_plasma'] = self.Lz_antenna = self.Lz_plasma_src = base_instruction[-1]['z_end']
                    self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = self.cfg['DOMAIN'].get('Lz_wall', 0.02)
                    self.instructions = self.antenna_grill.generate_mesh_instructions(z_start_position=self.Lz_wall)
                else:
                    self.cfg['DOMAIN']['Lz_plasma'] = self.Lz_plasma_src = 3. * self.lambda_para
                    self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = 0.
                    self.instructions = []

        self.cfg['DOMAIN']['Lz_tot'] = self.Lz_tot = self.Lz_plasma_src + 2.0 * self.Lz_wall + 2.0 * self.Lz_pml

        print(f'==== \n'
            f'Lx_plasma: {self.Lx_plasma:.2e}m,   Lx_pml: {self.Lx_pml:.2e}m,    Lx_tot: {self.Lx_tot:.2e}m\n'
            f'Lx_wg = {self.Lx_wg:.2e}m')
        print(f'Lz_antenna: {getattr(self, "Lz_antenna", 0):.2e}m, Lz_wall: {self.Lz_wall:.2e}m')
        print(f'Lz_plasma: {self.Lz_plasma_src:.2e}m, Lz_pml: {self.Lz_pml:.2e}m, Lz_tot: {self.Lz_tot:.2e}m')

        # Affichage mis à jour pour bien différencier Bord et Cœur
        print(f'==== \n'
            f'n_∥: {self.n_para:.1f},   λ_∥: {self.lambda_para:.2e}m \n'
            f'n_⟂⁺ (edge): {self.n_perp_p:.2f}, n_⟂⁺ (core): {np.abs(n_perp_eval):.2f}\n'
            f'λ_⟂⁺ (core): {self.lambda_perp_eval:.2e}m \n'
            f'n_⟂-:{self.n_perp_m:.2f}, λ_⟂⁻: {self.lambda_perp_m:.2e}m')

        # --- NOUVEAUTÉ : Résolution globale basée sur l'indice maximum strict ---
        n_index_meshing = max(np.abs(n_perp_eval), np.abs(self.n_perp_m), np.abs(self.n_para), 1.0)
        shortest_lambda = self.lambda0 / n_index_meshing

        # Le maxh_plasma garantit que même le coeur à haute densité est parfaitement résolu.
        self.maxh_plasma = shortest_lambda / self.cfg['DOMAIN']['ppw_medium']

        # print(f'==== \n'
        #     f'n_index_meshing: {n_index_meshing:.2f} \n'
        #     f'λ_meshing (shortest): {shortest_lambda:.2e}m \n'
        #     f'maxh_plasma: {self.maxh_plasma:.2e}m')

        # --- Create the geometry based on the previously recovered dimensions
        # Set simulation box (rect_plasma is equivalent to rect vacuum in dimensions size)
        rect_plasma_left = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_wall).Face()
        rect_plasma_src = occ.MoveTo(0, self.Lz_wall).Rectangle(self.Lx_plasma, self.Lz_plasma_src).Face()
        rect_plasma_right = occ.MoveTo(0, self.Lz_wall + self.Lz_plasma_src).Rectangle(self.Lx_plasma, self.Lz_wall).Face()
        # rect_plasma_left.maxh, rect_plasma_src, rect_plasma_right = self.maxh_plasma, self.maxh_plasma, self.maxh_plasma
        # Generate waveguide rectangles iff antenna is defined (not None)
        wg_faces = []
        if self.Lx_wg > 0:
            for inst in self.instructions:
                if inst['type'] in ['wg_active', 'wg_passive']:
                    width_wg = inst['z_end'] - inst['z_start']
                    face = occ.MoveTo(-self.Lx_wg, inst['z_start']).Rectangle(self.Lx_wg, width_wg).Face()
                    wg_faces.append(face)

        # Generate Geometry rectangles
        if self.geom_mode == "1D":
            ppw_pml = self.cfg['PML'].get('ppw_pml', 50)
            Sx_norm_max = np.sqrt(self.cfg['PML']['Sx_r']**2 + self.cfg['PML']['Sx_im']**2)
            maxh_pml_radial = shortest_lambda / (ppw_pml * Sx_norm_max)

            rect_pml_radial_left = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_wall).Face()
            rect_pml_radial_middle = occ.MoveTo(self.Lx_plasma, self.Lz_wall).Rectangle(self.Lx_pml, self.Lz_plasma_src).Face()
            rect_pml_radial_right = occ.MoveTo(self.Lx_plasma, self.Lz_wall + self.Lz_plasma_src).Rectangle(self.Lx_pml, self.Lz_wall).Face()
            faces = [rect_plasma_src, rect_pml_radial_middle]
            if self.Lz_wall > 0:
                faces += [rect_plasma_left, rect_plasma_right, rect_pml_radial_left, rect_pml_radial_right]
            domain = occ.Glue(faces + wg_faces)

            for e in domain.edges:
                if abs(e.center[0]) < 1e-6:
                    if self.Lz_wall < e.center[1] < (self.Lz_wall + self.Lz_plasma_src):
                        e.name = "bottom_source"
                    else:
                        e.name = "bottom_wall_pec"
                elif abs(e.center[0] - self.Lx_tot) < 1e-6:
                    e.name = "top_wall_pec"
                    e.maxh = max(maxh_pml_radial, 1e-4)

            for bot_edge in domain.edges:
                if abs(bot_edge.center[1]) < 1e-6:
                    bot_edge.name = "periodic_bot"
                    for top_edge in domain.edges:
                        if abs(top_edge.center[1] - self.Lz_plasma_src) < 1e-6 and abs(top_edge.center[0] - bot_edge.center[0]) < 1e-6:
                            top_edge.name = "periodic_top"
                            bot_edge.Identify(top_edge, "periodic", occ.IdentificationType.PERIODIC)
            # print('[DOMAIN 1D DEFINED]')

        else: # FULL_2D or VACUUM (with toroidal PMLs)
            if self.antenna_grill is None:
                # self.Lz_wall = 0.15 * self.Lz_plasma_src
                # self.Lz_plasma_src = self.Lz_plasma_src - (2.0 * self.Lz_wall)
                print(f'==== \n'
                f'Lz_wall: {self.Lz_wall:.2e}m \n'
                f'Lz_source: {self.Lz_plasma_src:.2e}m')

            ppw_pml = self.cfg['PML'].get('ppw_pml', 50)
            Sx_norm_max = np.sqrt(self.cfg['PML']['Sx_r']**2 + self.cfg['PML']['Sx_im']**2)
            maxh_pml_radial = shortest_lambda / (ppw_pml * Sx_norm_max)
            Sz_norm_max = np.sqrt(self.cfg['PML']['Sz_r']**2 + self.cfg['PML']['Sz_im']**2)
            maxh_pml_toroidal = shortest_lambda / (ppw_pml * Sz_norm_max)
            maxh_pml_corner = shortest_lambda / (ppw_pml * np.sqrt(Sx_norm_max**2 + Sz_norm_max**2))

            # print(f'==== \n'
            #     f'ppw_pml: {ppw_pml:.1f} \n'
            #     f'Sx_norm:{Sx_norm_max:.2e}, maxh_pml_radial: {maxh_pml_radial:.2e} \n'
            #     f'Sz_norm_max: {Sz_norm_max:.2e}, maxh_pml_toroidal: {maxh_pml_toroidal:.2e} \n'
            #     f'maxh_pml_corner: {maxh_pml_corner:.2e}')

            # 1. Build BARE geometry (NO NAMES, NO MAXH YET)
            rect_plasma_left = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_wall).Face()
            rect_plasma_src = occ.MoveTo(0, self.Lz_wall).Rectangle(self.Lx_plasma, self.Lz_plasma_src).Face()
            rect_plasma_right = occ.MoveTo(0, self.Lz_wall + self.Lz_plasma_src).Rectangle(self.Lx_plasma, self.Lz_wall).Face()

            rect_pml_radial_left = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_wall).Face()
            rect_pml_radial_middle = occ.MoveTo(self.Lx_plasma, self.Lz_wall).Rectangle(self.Lx_pml, self.Lz_plasma_src).Face()
            rect_pml_radial_right = occ.MoveTo(self.Lx_plasma, self.Lz_wall + self.Lz_plasma_src).Rectangle(self.Lx_pml, self.Lz_wall).Face()

            rect_pml_toroidal_left = occ.MoveTo(0, -self.Lz_pml).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            rect_pml_toroidal_right = occ.MoveTo(0, self.Lz_plasma_src + 2.0 * self.Lz_wall).Rectangle(self.Lx_plasma, self.Lz_pml).Face()

            rect_pml_corner_rad_left = occ.MoveTo(self.Lx_plasma, -self.Lz_pml).Rectangle(self.Lx_pml, self.Lz_pml).Face()
            rect_pml_corner_rad_right = occ.MoveTo(self.Lx_plasma, self.Lz_plasma_src + 2.0 * self.Lz_wall).Rectangle(self.Lx_pml, self.Lz_pml).Face()

            # 2. GLUE FIRST (Dissolves internal boundaries into a unified topology)
            domain = occ.Glue([rect_plasma_left, rect_plasma_src, rect_plasma_right,
                               rect_pml_radial_left, rect_pml_radial_middle, rect_pml_radial_right,
                               rect_pml_toroidal_left, rect_pml_toroidal_right,
                               rect_pml_corner_rad_left, rect_pml_corner_rad_right] + wg_faces)

            # print('[DOMAIN 2D DEFINED AND GLUED]')

            for e in domain.edges:
                c = e.center
                if self.Lx_wg > 0 and abs(c[0] - (-self.Lx_wg)) < 1e-6:
                    e.name = "bottom_source"
                elif self.Lx_wg > 0 and c[0] < -1e-6 and abs(c[0] - (-self.Lx_wg)) > 1e-6:
                    e.name = "bottom_wall_pec"
                elif abs(c[0]) < 1e-6:
                    if self.Lx_wg > 0:
                        is_opening = False
                        for inst in self.instructions:
                            if inst['type'] in ['wg_active', 'wg_passive']:
                                if inst['z_start'] - 1e-5 < c[1] < inst['z_end'] + 1e-5:
                                    is_opening = True
                                    break
                    else:
                        if self.Lz_wall < c[1] < (self.Lz_wall + self.Lz_plasma_src):
                            e.name = "bottom_source"
                        else:
                            e.name = "bottom_wall_pec"
                elif abs(c[0] - self.Lx_tot) < 1e-6:
                    e.name = "top_wall_pec"
                    e.maxh = maxh_pml_radial
                elif abs(c[1] - (-self.Lz_pml)) < 1e-6:
                    e.name = "left_wall_pec"
                    e.maxh = maxh_pml_toroidal
                elif abs(c[1] - (self.Lz_plasma_src + 2.0 * self.Lz_wall + self.Lz_pml)) < 1e-6:
                    e.name = "right_wall_pec"
                    e.maxh = maxh_pml_toroidal

        geo = occ.OCCGeometry(domain, dim=2)
        # Apply the global maximum size for the bulk plasma
        with TaskManager():
            self.mesh = Mesh(geo.GenerateMesh(maxh=self.maxh_plasma))
            self.mesh.Curve(3)
        print('[MESH GENERATED !]')
        return self.mesh


    def build_antenna_source_function(self):
        # Translates AntennaGrill instructions into a continuous NGSolve CoefficientFunction.

        if self.antenna_grill is None:
            # Fallback to the old single plane wave
            if self.box_medium == 'VACUUM':
                return self.cfg['WAVE']['E_inc'] * exp(1j * self.k0 * self.n_para * self.z)
            else:
                return self.cfg['WAVE']['E_inc'] * exp(1j * self.k0 * self.n_para * self.z)

        # Initialize an empty complex field
        Ez_inc_cf = CF(0.0 + 0.0j)

        # Build the piecewise function
        print(f'len(instruction): {len(self.instructions)}')
        for inst in self.instructions:
            # print(f"inst['type']: {inst['type']}")
            if inst['type'] == 'wg_active':
                z_start = inst['z_start']
                z_end = inst['z_end']
                E_val = inst['complex_E_field']

                #print(f'z_start: {z_start}, z_end: {z_end}, E_val: {E_val}')
            # We want: 1.0 IF (z > z_start AND z < z_end) ELSE 0.0
            # Condition 1: z - z_start > 0
            # Condition 2: z_end - z > 0
            # IfPos: If (z - z_start > 0) -> Evaluate (If z_end - z > 0) -> Return E_val

                is_inside_wg = IfPos(self.z - z_start, IfPos(z_end - self.z, 1.0, 0.0), 0.0)

                # Add this segment's field to the total function
                Ez_inc_cf += is_inside_wg * E_val
        return Ez_inc_cf

# =====================================================================
# PHYSICS IMPLEMENTATION - STIX TENSOR, B FIELD, etc...
# =====================================================================
    def build_physics_Stix_B_field(self) -> None:
        '''
        Function to gather every needed physical parameters to build the Stix tensor in cold plasma approximation
        that contain all the plasma/wave physics and geometry:
            - Compute arbitrary B field components relative to cartesian coordinates
            - Compute every general Stix tensor elements
            and return it as a native NGSolve CoefficientFunction
        '''
        if self.box_medium == "PLASMA":
            box_is_plasma = 1.0
        else:
            box_is_plasma = 0.0
        print(f'Lx_wg: {self.Lx_wg:.2e}m, wg_medium: {self.wg_medium}')
        if self.Lx_wg > 0 and self.wg_medium == "PLASMA":
            print(f'Waveguide medium is PLASMA, wg_is_plasma set to 1.0')
            wg_is_plasma = 1.0
        else:
            print(f'Waveguide medium is not PLASMA, wg_is_plasma set to 0.0')
            wg_is_plasma = 0.0

        # Mappage spatial rigoureux : Guide d'onde (x < 0) vs Boîte principale (x >= 0)
        is_plasma = IfPos(self.x, box_is_plasma, wg_is_plasma)
        is_vacuum = 1.0 - is_plasma

        theta_B, phi_B = 0.0, 0.0
        bx = np.sin(phi_B)
        by = np.cos(phi_B) * np.sin(theta_B)
        bz = np.cos(phi_B) * np.cos(theta_B)

        points = self.cfg['PLASMA']['ne_points'] # Exemple: [(0.0, 1e17), (0.01, 1e18), (0.05, 5e18)]

        # Le point final définit le début de la PML radiale (le cœur du plasma)
        x_core, ne_core = points[-1]

        # Étape A : On initialise la structure avec la valeur constante du cœur (pour x >= x_core)
        ne_plasma = CF(ne_core)

        # Étape B : On boucle à l'envers sur les segments pour imbriquer les structures conditionnelles IfPos
        for i in reversed(range(len(points) - 1)):
            x_i, ne_i = points[i]
            x_next, ne_next = points[i+1]

            # Calcul de la pente locale pour ce segment spécifique
            slope = (ne_next - ne_i) / (x_next - x_i)
            val_segment = ne_i + slope * (self.x - x_i)

            # Si x > x_next, on utilise le profil des segments plus profonds
            # Sinon (x <= x_next), on applique la rampe linéaire de ce segment
            ne_plasma = IfPos(self.x - x_next, ne_plasma, val_segment)

        # Étape C : Gestion de la zone des guides d'ondes (si x < 0)
        wg_is_plasma = 1.0 if (self.Lx_wg > 0 and self.wg_medium == "PLASMA") else 0.0
        ne_edge = points[0][1]

        # Enregistrement du profil complet sous forme de variable d'instance pour le plot
        self.ne_profile_cf = IfPos(self.x, box_is_plasma * ne_plasma, wg_is_plasma * ne_edge)
        w_pe2 = (self.ne_profile_cf * self.qe**2) / (self.me * self.eps0)
        w_pi2 = (self.ne_profile_cf * self.qe**2) / (self.mi * self.eps0)

        S_cf = 1.0 - w_pe2/(self.omega_LH**2 - self.Om_ce**2) - w_pi2/(self.omega_LH**2 - self.Om_ci**2)
        P_cf = 1.0 - w_pe2/self.omega_LH**2 - w_pi2/self.omega_LH**2
        D_cf = -(self.Om_ce * w_pe2)/(self.omega_LH*(self.omega_LH**2 - self.Om_ce**2)) + \
                (self.Om_ci * w_pi2)/(self.omega_LH*(self.omega_LH**2 - self.Om_ci**2))
        Q_stix = P_cf - S_cf

        self.K_xx = S_cf*(1 - bx**2) + P_cf*bx**2
        self.K_xy = 1j*D_cf*bz + Q_stix*bx*by
        self.K_xz = -1j*D_cf*by + Q_stix*bx*bz

        self.K_yx = -1j*D_cf*bz + Q_stix*by*bx
        self.K_yy = S_cf*(1 - by**2) + P_cf*by**2
        self.K_yz = 1j*D_cf*bx + Q_stix*by*bz

        self.K_zx = 1j*D_cf*by + Q_stix*bz*bx
        self.K_zy = -1j*D_cf*bx + Q_stix*bz*by
        self.K_zz = S_cf*(1 - bz**2) + P_cf*bz**2

        self.K_tensor = CF((self.K_xx, self.K_xy, self.K_xz,
                            self.K_yx, self.K_yy, self.K_yz,
                            self.K_zx, self.K_zy, self.K_zz), dims=(3,3))

    def get_port_admittance(self):
        """ Computes the robust 2x2 Reflected Admittance Tensor depending on the medium """
        Z0 = np.sqrt(self.mu0 / self.eps0)
        port_medium = self.wg_medium if self.Lx_wg > 0 else self.box_medium

        if port_medium == "PLASMA":
            Py_S = (1j * self.D * (self.n_perp_p**2 - self.P)) / (self.n_perp_p * self.n_para * (self.n_perp_p**2 - self.S))
            Py_F = (1j * self.D * (self.n_perp_m**2 - self.P)) / (self.n_perp_m * self.n_para * (self.n_perp_m**2 - self.S))

            denom = Z0 * (Py_S - Py_F)
            Y_11_inc = (-self.P/self.n_perp_p + self.P/self.n_perp_m) / denom
            Y_12_inc = (self.P*Py_F/self.n_perp_p - self.P*Py_S/self.n_perp_m) / denom
            Y_21_inc = (self.n_perp_p*Py_S - self.n_perp_m*Py_F) / denom
            Y_22_inc = (Py_S*Py_F*(self.n_perp_m - self.n_perp_p)) / denom

            Y_11_ref = Y_11_inc
            Y_12_ref = -Y_12_inc
            Y_21_ref = -Y_21_inc
            Y_22_ref = Y_22_inc

        else: # VACUUM (Onde TM pure et rigoureuse)
            n_perp_vac_port = 1.0 + 0.0j if self.Lx_wg > 0 else np.sqrt(complex(1.0 - self.n_para**2))
            Y_11_ref, Y_22_ref = 0.0 + 0.0j, 0.0 + 0.0j
            Y_12_ref = 1.0 / (n_perp_vac_port * Z0)
            Y_21_ref = -1.0 / (n_perp_vac_port * Z0)

        return Y_11_ref, Y_12_ref, Y_21_ref, Y_22_ref

    def get_incident_fields(self, Ez_inc_val):
        """ Computes the incident E and H fields strictly adapted to the medium """
        Z0 = np.sqrt(self.mu0 / self.eps0)
        port_medium = self.wg_medium if self.Lx_wg > 0 else self.box_medium

        if port_medium == "PLASMA":
            Py_S = (1j * self.D * (self.n_perp_p**2 - self.P)) / (self.n_perp_p * self.n_para * (self.n_perp_p**2 - self.S))
            Ey_inc = Py_S * Ez_inc_val
            Ez_inc = Ez_inc_val
            Hy_inc = (-self.P / (self.n_perp_p * Z0)) * Ez_inc_val
            Hz_inc = ((self.n_perp_p * Py_S) / Z0) * Ez_inc_val

        else: # VACUUM
            n_perp_vac_port = 1.0 + 0.0j if self.Lx_wg > 0 else np.sqrt(complex(1.0 - self.n_para**2))
            print(f'n_perp_vac_port: {n_perp_vac_port:.2f}')
            Ey_inc = CF(0.0 + 0.0j)
            Ez_inc = Ez_inc_val
            Hy_inc = -(1.0 / (n_perp_vac_port*Z0)) * Ez_inc_val
            Hz_inc = CF(0.0 + 0.0j)

        return Ey_inc, Ez_inc, Hy_inc, Hz_inc

# =====================================================================
# SOLVE HELMHOLTZ 3D IN 2D BOX DOMAIN
# =====================================================================
    def solve_helmholtz_2DHcurl_1DH1_with_pml(self, mesh, geom_mode, box_medium):
        '''
        Solves the Weak Form using standard (Ex, Ey, Ez) coordinate mapping: (Ex = E_3D[0] = Radial, Ey = E_3D[1] = Poloidal, Ez = E_3D[2] = Toroidal)
            - Set Hcurl finite element space (fes) for in plane E field components (Ex, Ez): Hcurl compute E field and needles between meshpoints
            H1 (fes) compute out of 2D plane E field component (Ey) and force the field vector continuity in every space directions
            -
        '''
        max_amr_steps = 5
        tolerance = 1e-4
        amr_history = {
        'ndofs': [],
        'gamma_R': [],
        'gamma_T': [],
        'power_error': []
        }

        if geom_mode == "1D":
            dirichlet_bnds = "top_wall_pec"

        else: dirichlet_bnds = "left_wall_pec|right_wall_pec|top_wall_pec|bottom_wall_pec"

        fes_plane = HCurl(mesh, order=3, complex=True, dirichlet=dirichlet_bnds)
        fes_outplane = H1(mesh, order=3, complex=True, dirichlet=dirichlet_bnds)


        if geom_mode == "1D":
            fes_plane, fes_outplane = Periodic(fes_plane), Periodic(fes_outplane)

        self.fes = fes_plane * fes_outplane
        fes_flux = H1(self.mesh, order=3, complex=True, dim=3)
        print(f'==== \n'
            f'#DoFs = {self.fes.ndof}')

        # --- Dynamic PML Stretching ---
        Sx_r, Sx_im, px = self.cfg['PML']['Sx_r'], self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
        Sz_r, Sz_im, pz = self.cfg['PML']['Sz_r'], self.cfg['PML']['Sz_im'], self.cfg['PML']['pz']

        sign_x = 1.0 if self.box_medium == "VACUUM" else -1.0
        sign_z = 1.0

        Stretch_x = 1.0 + (Sx_r - 1.0 + 1j * sign_x * Sx_im) * \
                    IfPos(self.x - self.Lx_plasma, ((self.x - self.Lx_plasma) / self.Lx_pml)**px, 0.0)

        if geom_mode == "1D":
            Stretch_z = CF(1.0)
        else:
            Stretch_z = 1.0 + (Sz_r - 1.0 + 1j * sign_z * Sz_im) * \
                        IfPos(-self.z, (-self.z / self.Lz_pml)**pz, \
                        IfPos(self.z - (self.Lz_plasma_src+ 2*self.Lz_wall), ((self.z - (self.Lz_plasma_src+ 2*self.Lz_wall)) / self.Lz_pml)**pz, 0.0))
        # print(f'Stretch_z (RADIAL_ONLY) : {Stretch_z}')
        self.pml_tensor = CF((Stretch_x / Stretch_z, 0.0, 0.0,
                              0.0, 1.0/(Stretch_x * Stretch_z), 0.0,
                              0.0, 0.0, Stretch_z / Stretch_x), dims=(3,3))

        self.eff_eps_tensor = CF((
            self.K_xx * (Stretch_z / Stretch_x), self.K_xy * (Stretch_z / Stretch_x), self.K_xz * (Stretch_z / Stretch_x),
            self.K_yx * (Stretch_x * Stretch_z), self.K_yy * (Stretch_x * Stretch_z), self.K_yz * (Stretch_x * Stretch_z),
            self.K_zx * (Stretch_x / Stretch_z), self.K_zy * (Stretch_x / Stretch_z), self.K_zz * (Stretch_x / Stretch_z)
        ), dims=(3,3))

        for step in range(max_amr_steps):
            print(f"\n--- AMR Iteration {step+1}/{max_amr_steps} ---")
            print(f"Degrees of Freedom: {self.fes.ndof}")
            self.fes.Update()
            fes_flux.Update()

            # --- Vector Assembly ---
            self.E_field = GridFunction(self.fes)
            E_plane, E_outplane = self.fes.TrialFunction()
            v_plane, v_outplane = self.fes.TestFunction()

            E_3D = CF((E_plane[0], E_outplane, E_plane[1]))
            v_3D = CF((v_plane[0], v_outplane, v_plane[1]))

            curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0] ))
            curl_v_3D = CF(( -grad(v_outplane)[1], -curl(v_plane), grad(v_outplane)[0] ))

            Y_11_ref, Y_12_ref, Y_21_ref, Y_22_ref = self.get_port_admittance()

            # bi-linear form:
            a = BilinearForm(self.fes)
            # volume intergral (unknown) terms:
            a += (self.pml_tensor * curl_E_3D * curl_v_3D - \
                self.k0**2 * self.eff_eps_tensor * E_3D * v_3D)*dx
            # boundary intergral (unknown) terms:
            Ez_trace, Ey_trace = E_plane.Trace()[1], E_outplane.Trace()
            vz_trace, vy_trace = v_plane.Trace()[1], v_outplane.Trace()
            a += 2j * self.omega_LH * self.mu0 * ((Y_21_ref * Ey_trace + Y_22_ref * Ez_trace) * vy_trace - \
                (Y_11_ref * Ey_trace + Y_12_ref * Ez_trace) * vz_trace) * ds("bottom_source")

            with TaskManager():
                a.Assemble()
                f=LinearForm(self.fes)

                Ez_inc_spatial = self.build_antenna_source_function()
                Ey_inc, Ez_inc, Hy_inc, Hz_inc = self.get_incident_fields(Ez_inc_spatial)

                Ay = (Y_11_ref * Ey_inc + Y_12_ref * Ez_inc) - Hy_inc
                Az = (Y_21_ref * Ey_inc + Y_22_ref * Ez_inc) - Hz_inc
                f+= 1j * self.omega_LH * self.mu0 * (Az * vy_trace - Ay * vz_trace) * ds("bottom_source")
                f.Assemble()

                # print("--- Solving the 3D vector linear system ---")
                res = f.vec.CreateVector()
                res.data = f.vec - a.mat * self.E_field.vec
                inv = a.mat.Inverse(freedofs=self.fes.FreeDofs())
                self.E_field.vec.data += inv * res

            # =====================================================================
            # RIGOROUS GAMMA REFLECTION COMPUTATION
            # =====================================================================
            E_xz, Ey = self.E_field.components[0], self.E_field.components[1]
            Ex, Ez = E_xz[0], E_xz[1]
            E_tot_norm = sqrt(Ex*Conj(Ex) + Ey*Conj(Ey) + Ez*Conj(Ez))

            window_size_radial = 1.5 * self.cfg['WAVE']['lambda0'] / np.abs(self.n_perp_p.real)
            window_size_toroidal = min(1.5 * self.lambda_para, 0.5 * self.Lz_plasma_src)

            # FIXED: Total physical plasma length including walls
            z_max_plasma = self.Lz_plasma_src + 2.0 * self.Lz_wall

            print(f'==== \n'
                f'w_size_R: {window_size_radial:.4e}m,    w_size_T: {window_size_toroidal:.4e}m')

            # ---------------------------------------------------------------------
            # 1. RADIAL SWR (Right Wall: Reflection along X-axis)
            # ---------------------------------------------------------------------
            x_target_R = self.Lx_plasma * 0.95

            # FIXED: Sweep across the entire valid Z domain
            z_sweep_R = np.linspace(0.01 * z_max_plasma, 0.99 * z_max_plasma, 1000)
            mips_target_R = self.mesh(np.full_like(z_sweep_R, x_target_R), z_sweep_R)

            E_target_R, valid_z_R = [], []
            for z, mip in zip(z_sweep_R, mips_target_R):
                try:
                    val = E_tot_norm(mip).real
                    if not np.isnan(val):
                        E_target_R.append(val)
                        valid_z_R.append(z)
                except: pass

            if len(E_target_R) > 0:
                peak_z_R = valid_z_R[np.argmax(E_target_R)]
            else:
                peak_z_R = z_max_plasma / 2.0

            x_sweep_R = np.linspace(0.01 * self.Lx_plasma, 0.99 * self.Lx_plasma, 1000)
            mips_measure_R = self.mesh(x_sweep_R, np.full_like(x_sweep_R, peak_z_R))

            E_measure_R, valid_x_R = [], []
            for x, mip in zip(x_sweep_R, mips_measure_R):
                try:
                    val = E_tot_norm(mip).real
                    if not np.isnan(val):
                        E_measure_R.append(val)
                        valid_x_R.append(x)
                except: pass

            valid_x_R = np.array(valid_x_R)
            E_measure_R = np.array(E_measure_R)

            mask_R = (valid_x_R <= x_target_R) & (valid_x_R >= x_target_R - window_size_radial)
            E_window_R = E_measure_R[mask_R]

            if len(E_window_R) > 5:
                E_min_R = max(np.min(E_window_R), 1e-12)
                SWR_R = max(np.max(E_window_R) / E_min_R, 1.000001)
                Gamma_R = (SWR_R - 1.0) / (SWR_R + 1.0)
                print(f'Gamma_R (SWR) = {Gamma_R:.3e}')
            else:
                Gamma_R = 1.0

            # ---------------------------------------------------------------------
            # 2. TOROIDAL SWR (Top/Bottom Wall: Reflection along Z-axis)
            # ---------------------------------------------------------------------
            if self.n_para.real >= 0:
                z_target_T = z_max_plasma * 0.95
            else:
                z_target_T = z_max_plasma * 0.05

            x_sweep_T = np.linspace(0.01 * self.Lx_plasma, 0.99 * self.Lx_plasma, 1000)
            mips_target_T = self.mesh(x_sweep_T, np.full_like(x_sweep_T, z_target_T))

            E_target_T, valid_x_T = [], []
            for x, mip in zip(x_sweep_T, mips_target_T):
                try:
                    val = E_tot_norm(mip).real
                    if not np.isnan(val):
                        E_target_T.append(val)
                        valid_x_T.append(x)
                except: pass

            if len(E_target_T) > 0 and np.max(E_target_T) > 1e-3 * self.cfg['WAVE']['E_inc']:
                peak_x_T = valid_x_T[np.argmax(E_target_T)]
            else:
                peak_x_T = self.Lx_plasma / 2.0
                print("[WARNING] Resonance cone likely missed the toroidal wall. Gamma_T measurements may be noise.")

            # FIXED: Measure line now spans the whole domain
            z_sweep_T = np.linspace(0.01 * z_max_plasma, 0.99 * z_max_plasma, 1000)
            mips_measure_T = self.mesh(np.full_like(z_sweep_T, peak_x_T), z_sweep_T)

            E_measure_T, valid_z_T = [], []
            for z, mip in zip(z_sweep_T, mips_measure_T):
                try:
                    val = E_tot_norm(mip).real
                    if not np.isnan(val):
                        E_measure_T.append(val)
                        valid_z_T.append(z)
                except: pass

            valid_z_T = np.array(valid_z_T)
            E_measure_T = np.array(E_measure_T)

            if self.n_para.real >= 0:
                mask_T = (valid_z_T <= z_target_T) & (valid_z_T >= z_target_T - window_size_toroidal)
            else:
                mask_T = (valid_z_T >= z_target_T) & (valid_z_T <= z_target_T + window_size_toroidal)

            E_window_T = E_measure_T[mask_T]

            # Safety envelope check
            if len(E_window_T) > 5 and np.max(E_window_T) > 1e-4:
                SWR_T = max(np.max(E_window_T) / np.max([np.min(E_window_T), 1e-12]), 1.000001)
                Gamma_T = (SWR_T - 1.0) / (SWR_T + 1.0)
                print(f'Gamma_T (SWR) = {Gamma_T:.3e}')
            else:
                Gamma_T = 0.0
                print("Gamma_T FAILED: Window missed, too small, or beam exited radially.")


            # ---------------------------------------------------------------------
            # POYNTING VECTOR COMPUTATION
            # ---------------------------------------------------------------------
            print("\n--- Poynting Flux & Energy Conservation ---")
            self.mu0 = self.cfg['CONST']['mu0']

            E_plane_gf = self.E_field.components[0]
            E_outplane_gf = self.E_field.components[1]

            E_sol_3D = CF((E_plane_gf[0], E_outplane_gf, E_plane_gf[1]))

            # FIXED: Use explicit native 2D curl() for HCurl to get the correct d(Ex)/dz - d(Ez)/dx
            curl_E_sol_3D = CF(( -E_outplane_gf.Deriv()[1],
                                -curl(E_plane_gf),
                                E_outplane_gf.Deriv()[0] ))

            H_sol_3D = curl_E_sol_3D / (1j * self.omega_LH * self.mu0)

            S_vec = 0.5 * Cross(E_sol_3D, Conj(H_sol_3D)).real
            S_x_cf = S_vec[0]
            S_z_cf = S_vec[2]

            def integrate_flux(cf, x_vals, z_vals, axis):
                vals, coords = [], []
                for xi, zi in zip(x_vals, z_vals):
                    try:
                        mip = self.mesh(xi, zi)
                        vals.append(cf(mip))
                        coords.append(zi if axis=='z' else xi)
                    except Exception: pass
                if len(coords) > 1:
                    return np.trapezoid(vals, x=coords)
                return 0.0

            N_pts = 1000
            x_limits = np.linspace(0.0, self.Lx_plasma, N_pts)

            # FIXED: Integrate across the ENTIRE physical plasma boundary
            z_limits = np.linspace(0.0, z_max_plasma, N_pts)

            P_in_net = integrate_flux(S_x_cf, np.full_like(z_limits, 0.0), z_limits, 'z')
            P_out_Radial = integrate_flux(S_x_cf, np.full_like(z_limits, self.Lx_plasma), z_limits, 'z')

            if self.geom_mode == "2D":
                # FIXED: Target the actual Toroidal walls
                P_out_Toroidal_Right = integrate_flux(S_z_cf, x_limits, np.full_like(x_limits, z_max_plasma), 'x')
                P_out_Toroidal_Left = -integrate_flux(S_z_cf, x_limits, np.full_like(x_limits, 0.0), 'x')
            else:
                P_out_Toroidal_Right, P_out_Toroidal_Left = 0.0, 0.0

            total_power_leaving_plasma = P_out_Radial + P_out_Toroidal_Right + P_out_Toroidal_Left

            print(f"Net Power Injected (Antenna): {P_in_net:.4e} W/m")
            print(f"Net Power Exiting to PMLs:    {total_power_leaving_plasma:.4e} W/m")

            power_error_plasma = abs(P_in_net - total_power_leaving_plasma) / max(abs(P_in_net), 1e-12)
            print(f"Plasma Bulk Conservation Error: {power_error_plasma * 100:.2e} %")
            # =====================================================================
            # JACQUOT 2013 :
            # =====================================================================
            # 1. Theoretical Amplitude Reflection (eta_pred)
            k_perp_real = np.abs(self.n_perp_p.real) * self.k0
            k_para_real = np.abs(self.n_para.real) * self.k0

            Sx_im, px = self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
            Sz_im, pz = self.cfg['PML']['Sz_im'], self.cfg['PML']['pz']

            # Factor 2.0 because eta is the AMPLITUDE reflection coefficient (|Gamma|)
            eta_pred_R = np.exp(-2.0 * k_perp_real * Sx_im * self.Lx_pml / (px + 1.0))
            eta_pred_T = np.exp(-2.0 * k_para_real * Sz_im * self.Lz_pml / (pz + 1.0))

            eta_sim_R = Gamma_R
            eta_sim_T = Gamma_T

            # 2. Power flux profile inside the Radial PML
            x_pml_sweep = np.linspace(self.Lx_plasma, self.Lx_tot, 150)
            Px_pml_profile = []
            z_measure = self.Lz_plasma_src / 2.0

            for xi in x_pml_sweep:
                try:
                    mip = self.mesh(xi, z_measure)
                    Px_pml_profile.append(S_x_cf(mip).real)
                except:
                    Px_pml_profile.append(0.0)

            Px_pml_profile = np.array(Px_pml_profile)
            # Normalize the profile by the power at the PML entrance (to start at 1.0)
            if np.abs(Px_pml_profile[0]) > 1e-15:
                Px_pml_profile_norm = Px_pml_profile / Px_pml_profile[0]
            else:
                Px_pml_profile_norm = Px_pml_profile

            amr_history['ndofs'].append(self.fes.ndof)
            amr_history['gamma_R'].append(Gamma_R)
            amr_history['gamma_T'].append(Gamma_T)
            amr_history['power_error'].append(power_error_plasma)


            # Convergence Check
            if step > 0:
                delta_gamma = abs(amr_history['gamma_R'][-1] - amr_history['gamma_R'][-2])
                print(f"Delta Gamma_R: {delta_gamma:.3e}")
                if delta_gamma < tolerance:
                    print("--- AMR Converged ---")
                    break

            if step == max_amr_steps - 1:
                break # Do not refine on the final step

            # 4. Compute ZZ Error Estimator
            # Reconstruct H_sol_3D as you did in your Poynting vector calculation
            E_plane_gf, E_outplane_gf = self.E_field.components[0], self.E_field.components[1]
            curl_E_sol = CF(( -E_outplane_gf.Deriv()[1], -curl(E_plane_gf), E_outplane_gf.Deriv()[0] ))
            H_sol_3D = curl_E_sol / (1j * self.omega_LH * self.mu0)

            # Project H_sol_3D into continuous HDiv space
            u_flux, v_flux = fes_flux.TnT()
            a_flux = BilinearForm(u_flux * v_flux * dx).Assemble()
            f_flux = LinearForm(H_sol_3D * v_flux * dx).Assemble()

            gf_flux_smooth = GridFunction(fes_flux)
            gf_flux_smooth.vec.data = a_flux.mat.Inverse(freedofs=fes_flux.FreeDofs()) * f_flux.vec

            is_physical_plasma = IfPos(self.Lx_plasma - self.x, 1.0, 0.0) * \
                                 IfPos(self.z, 1.0, 0.0) * \
                                 IfPos((self.Lz_plasma_src + 2.0 * self.Lz_wall) - self.z, 1.0, 0.0)

            # is_physical_wg = 1.0 if inside the active/passive waveguides (x < 0)
            is_physical_wg = IfPos(-self.x, 1.0, 0.0)

            physical_mask = is_physical_plasma + is_physical_wg

            # Compute element-wise L2 error ONLY in the physical domain
            err_cf = physical_mask * Norm(H_sol_3D - gf_flux_smooth)**2
            elerr = Integrate(err_cf, self.mesh, element_wise=True)

            # 5. Mark Elements and Refine (Dörfler marking)
            max_err = max(elerr)
            for el in self.mesh.Elements():
                self.mesh.SetRefinementFlag(el, elerr[el.nr] > 0.25 * max_err)

            self.mesh.Refine()
        # ---------------------------------------------------------------------
        # EXPORT DIAGNOSTIC DATA FOR PLOTTING
        # ---------------------------------------------------------------------
        diag_data = {
            'x_target_R': x_target_R, 'peak_z_R': peak_z_R, 'window_size_radial': window_size_radial,
            'z_target_T': z_target_T, 'peak_x_T': peak_x_T, 'window_size_toroidal': window_size_toroidal,
            'n_para': self.n_para,
            'eta_sim_R': Gamma_R, 'eta_pred_R': eta_pred_R,
            'eta_sim_T': Gamma_T, 'eta_pred_T': eta_pred_T,
            'x_pml_sweep': x_pml_sweep,
            'Px_pml_profile_norm': Px_pml_profile_norm,
            'lambda_perp_real': (2.0 * np.pi) / k_perp_real,

            'P_in_net': P_in_net,
            'P_out_Radial': P_out_Radial,
            'P_out_Toroidal': P_out_Toroidal_Right + P_out_Toroidal_Left,
            'power_error_plasma': power_error_plasma
        }

        return self.E_field, Gamma_R, Gamma_T, diag_data, amr_history



    # =====================================================================
    # ALOHA BENCHMARK DIAGNOSTICS SUITE
    # =====================================================================

    def compute_waveguide_S_parameters(self):
        """
        Computes the complex reflection coefficient (Gamma) for each active waveguide.
        Extracts the field slightly inside the waveguide to avoid numerical singularities
        exactly on the boundary condition.
        """
        print("\n--- Extracting Waveguide S-Parameters (ALOHA Benchmark) ---")
        if self.antenna_grill is None or self.Lx_wg <= 0:
            print("[!] No physical waveguides defined. Skipping S-parameter extraction.")
            return {}

        # Evaluate slightly inside the waveguide to avoid PEC boundary singularities
        x_eval_wg = -self.Lx_wg + 1e-4
        gamma_dict = {}
        wg_counter = 1

        for idx, inst in enumerate(self.instructions):
            if inst['type'] == 'wg_active':
                z_start = inst['z_start']
                z_end = inst['z_end']
                E_inc_val = inst['complex_E_field']

                # Create a high-resolution sampling line across the waveguide gap
                # Retracting slightly from the metal septa walls (10 microns)
                z_wg = np.linspace(z_start + 1e-5, z_end - 1e-5, 500)

                E_tot = []
                for z in z_wg:
                    try:
                        mip = self.mesh(x_eval_wg, z)
                        # Extract Ez (Component 0 is in-plane, index 1 is z)
                        E_tot.append(self.E_field.components[0][1](mip))
                    except Exception:
                        E_tot.append(0.0 + 0.0j)

                E_tot = np.array(E_tot, dtype=complex)

                # Evaluate the incident field function numerically to ensure phase consistency
                E_inc_cf = self.build_antenna_source_function()
                E_inc_arr = []
                for z in z_wg:
                    try:
                        mip = self.mesh(x_eval_wg, z)
                        E_inc_arr.append(E_inc_cf(mip))
                    except Exception:
                        E_inc_arr.append(0.0 + 0.0j)
                E_inc_arr = np.array(E_inc_arr, dtype=complex)

                # Modal extraction integral:
                # Gamma = Integral( (E_tot - E_inc) * E_inc^* ) / Integral( |E_inc|^2 )
                numerator = np.trapezoid((E_tot - E_inc_arr) * np.conj(E_inc_arr), x=z_wg)
                denominator = np.trapezoid(E_inc_arr * np.conj(E_inc_arr), x=z_wg)

                if abs(denominator) > 1e-12:
                    gamma_i = numerator / denominator
                else:
                    gamma_i = 0.0 + 0.0j
                    print(f"[!] Warning: Denominator zero for WG at z={z_start:.3f}")

                gamma_dict[f'wg_{wg_counter}'] = {
                    'z_center': (z_start + z_end) / 2.0,
                    'Gamma_complex': gamma_i,
                    'Power_Reflectivity': np.abs(gamma_i)**2,
                    'Phase_deg': np.degrees(np.angle(gamma_i))
                }

                print(f"  -> WG_{wg_counter} | |Gamma|^2 = {np.abs(gamma_i)**2:.4f} | Phase = {np.degrees(np.angle(gamma_i)):.1f}°")
                wg_counter += 1
        return gamma_dict

    def extract_tangential_aperture_fields(self, x_target=0.0, num_points=4000):
        """
        Extracts Ex, Ey, Ez along the toroidal direction (z) at a specified depth (x).
        Default x=0.0 represents the plasma-antenna interface (aperture).
        """
        print(f"\n--- Extracting Field Profile at x={x_target:.4f}m ---")

        # We only want the physical plasma span, excluding toroidal PMLs
        z_max_plasma = self.Lz_plasma_src + 2.0 * self.Lz_wall
        z_sweep = np.linspace(0.0, z_max_plasma, num_points)

        Ex, Ey, Ez = [], [], []

        for z in z_sweep:
            try:
                mip = self.mesh(x_target, z)
                Ex.append(self.E_field.components[0][0](mip)) # In-plane, radial
                Ey.append(self.E_field.components[1](mip))    # Out-of-plane, poloidal
                Ez.append(self.E_field.components[0][1](mip)) # In-plane, toroidal
            except Exception:
                Ex.append(0.0 + 0.0j)
                Ey.append(0.0 + 0.0j)
                Ez.append(0.0 + 0.0j)

        Ex = np.array(Ex, dtype=complex)
        Ey = np.array(Ey, dtype=complex)
        Ez = np.array(Ez, dtype=complex)
        E_norm = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)

        field_data = {
            'z_coords': z_sweep,
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'E_norm': E_norm
        }
        return field_data

    def compute_aloha_normalized_spectrum(self, field_data, P_coupled_net):
        """
        Takes the z-profile data and returns the absolute power density spectrum
        dP/dn_para scaled to match the net injected Poynting flux.
        """
        print("\n--- Computing Normalized Power Spectrum (ALOHA Benchmark) ---")

        z_sweep = field_data['z_coords']
        Ez = field_data['Ez']
        dz = z_sweep[1] - z_sweep[0]
        num_points = len(z_sweep)

        # Spatial FFT
        Ez_fft = np.fft.fftshift(np.fft.fft(Ez, 8*num_points)) * dz

        # Wavevector and Refractive Index Arrays
        k_z = np.fft.fftshift(np.fft.fftfreq(8*num_points, d=dz)) * 2.0 * np.pi
        n_para = k_z / self.k0

        # Base Power Spectrum Shape: |E_z(n_para)|^2
        power_spectrum = np.abs(Ez_fft)**2

        # Rigorous Normalization: Integral of Spectrum = P_coupled_net
        dn_para = n_para[1] - n_para[0]
        current_integral = np.trapezoid(power_spectrum, dx=dn_para)

        if current_integral > 1e-15:
            # Scale to yield watts per unit of n_parallel
            normalized_power_spectrum = power_spectrum * (P_coupled_net / current_integral)
        else:
            normalized_power_spectrum = power_spectrum
            print("[!] Spectrum integral is zero. Normalization failed.")

        return n_para, normalized_power_spectrum


def export_mesh_for_paper(mesh, E_field, filename="mesh_output"):
    """
    Exports the mesh and the electric field magnitude to VTK.
    Open this in ParaView, apply a 'Wireframe' representation to see the AMR clustering.
    """
    vtk = VTKOutput(ma=mesh,
                    coefs=[Norm(E_field.components[0])],
                    names=["E_norm"],
                    filename=filename,
                    subdivision=2) # Subdivision creates smoother visualizations for higher order elements
    vtk.Do()
    print(f"VTK mesh exported to {filename}.vtu. Open in ParaView.")