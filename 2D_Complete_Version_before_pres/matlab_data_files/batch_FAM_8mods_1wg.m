
clear all;

% on veut faire un scan en densite sur les valeurs suivantes :
dens = [1, 1.2, 2, 3, 4, 5, 6, 8, 10, 20, 30, 40]*1e17;

scen{1} = 'swan_0_FAM_8mods_1wg_sc';


for id_scen = 1:length(scen)
    disp(['-----------', num2str(id_scen),'/',num2str(length(scen)),'------']);
    % on charge un scenario "type". 
    % NB : On s'est assure au prealable que tous les parametres autre que
    % la densite sont OK.
    sc = eval(scen{id_scen});
    % on reproduit ce scenario pour le nombre de valeur de densite:
    sc = repmat(sc, length(dens), 1);
    
    % on definit la valeur de densite ne0 dans les scenario 
    for id=1:length(dens)
        sc(id).plasma.ne0 = dens(id);
    end  
    
    % on lance ALOHA sur l'ensemble de ces memes scenarios 'type' 
    sc = aloha_scenario(sc);
    
    % on sauve le resultat d'un type de scenario
    aloha_scenario_save(sc, 'saved_batch_8mods_1wg.mat');

end % id_scen

