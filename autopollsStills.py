import subprocess
import os
import sys
import shutil
import cv2
import glob
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from keras.models import load_model
from ultralytics import YOLO

##################
detectModel = YOLO('/home/apis/Desktop/pcamData/pcamVids/HTC/Bernauer/YOLOV8_bee_detectors/YOLOv8l/best.pt')
detectionThres = 0.25
#####################
CATEGORIES = [
"Acamptopoeum", "Agapostemon", "Agapostemon_splendens", "Agapostemon_virescens", "Alloscirtetica", "Amegilla", "Amphylaeus",
"Ancylandrena", "Andrena", "Andrena_barbilabris", "Andrena_bicolor", "Andrena_cineraria", "Andrena_clarkella",
"Andrena_denticulata", "Andrena_dorsata", "Andrena_dunningi", "Andrena_erigeniae", "Andrena_flavipes", "Andrena_florea",
"Andrena_fulva", "Andrena_haemorrhoa", "Andrena_hattorfiana", "Andrena_hirticincta", "Andrena_milwaukeensis",
"Andrena_nigroaenea", "Andrena_nitida", "Andrena_nubecula", "Andrena_prunorum",
"Andrena_vaga", "Andrena_ventralis", "Andrena_wilkella", "Anthidiellum", "Anthidium", "Anthidium_florentinum",
"Anthidium_maculosum", "Anthidium_manicatum", "Anthidium_oblongatum", "Anthidium_punctatum", "Anthidium_septemspinosum",
"Anthophora", "Anthophora_abrupta", "Anthophora_bimaculata", "Anthophora_bomboides", "Anthophora_californica",
"Anthophora_furcata", "Anthophora_pacifica", "Anthophora_plumipes", "Anthophora_quadrimaculata", "Anthophora_retusa",
"Anthophora_terminalis", "Anthophora_urbana", "Anthophora_villosula", "Anthophorula", "Apis",
"Apis_cerana", "Apis_dorsata", "Apis_florea", "Apis_laboriosa", "Apis_mellifera", "Ashmeadiella", "Augochlora", "Augochlorella",
"Augochloropsis", "Bombus", "Bombus_affinis", "Bombus_alpinus", "Bombus_appositus", "Bombus_ardens", "Bombus_argillaceus",
"Bombus_asiaticus", "Bombus_atripes", "Bombus_auricomus", "Bombus_balteatus", "Bombus_barbutellus", "Bombus_beaticola",
"Bombus_bellicosus", "Bombus_bicoloratus", "Bombus_bifarius", "Bombus_bimaculatus", "Bombus_bohemicus", "Bombus_borealis",
"Bombus_brasiliensis", "Bombus_californicus", "Bombus_caliginosus", "Bombus_campestris", "Bombus_centralis",
"Bombus_cingulatus", "Bombus_citrinus", "Bombus_coccineus", "Bombus_confusus", "Bombus_consobrinus", "Bombus_crotchii",
"Bombus_cryptarum", "Bombus_cullumanus", "Bombus_dahlbomii", "Bombus_deuteronymus", "Bombus_distinguendus", "Bombus_diversus",
"Bombus_ephippiatus", "Bombus_eximius", "Bombus_fervidus", "Bombus_flavidus", "Bombus_flavifrons", "Bombus_formosellus",
"Bombus_fragrans", "Bombus_fraternus", "Bombus_frigidus", "Bombus_funebris", "Bombus_griseocollis", "Bombus_haematurus",
"Bombus_hedini", "Bombus_honshuensis", "Bombus_hortorum", "Bombus_hortulanus", "Bombus_humilis", "Bombus_huntii",
"Bombus_hyperboreus", "Bombus_hypnorum", "Bombus_hypocrita", "Bombus_ignitus", "Bombus_impatiens", "Bombus_insularis",
"Bombus_jonellus", "Bombus_kirbiellus", "Bombus_koreanus", "Bombus_laesus", "Bombus_lapidarius", "Bombus_lapponicus",
"Bombus_lucorum", "Bombus_magnus", "Bombus_mckayi", "Bombus_medius", "Bombus_melanopygus", "Bombus_melanurus",
"Bombus_mesomelas", "Bombus_mexicanus", "Bombus_mixtus", "Bombus_modestus", "Bombus_monticola", "Bombus_morio",
"Bombus_morrisoni", "Bombus_muscorum", "Bombus_nevadensis", "Bombus_niveatus", "Bombus_norvegicus", "Bombus_occidentalis",
"Bombus_opifex", "Bombus_opulentus", "Bombus_pascuorum", "Bombus_patagiatus", "Bombus_pauloensis", "Bombus_pensylvanicus",
"Bombus_perplexus", "Bombus_picipes", "Bombus_pratorum", "Bombus_pseudobaicalenis", "Bombus_pullatus", "Bombus_pyrenaeus",
"Bombus_pyrosoma", "Bombus_pyrrhopygus", "Bombus_quadricolor", "Bombus_robustus", "Bombus_rubicundus", "Bombus_ruderarius",
"Bombus_ruderatus", "Bombus_rufocinctus", "Bombus_rupestris", "Bombus_schrencki", "Bombus_semenoviellus", "Bombus_sichelii",
"Bombus_sitkensis", "Bombus_sonani", "Bombus_soroeensis", "Bombus_sporadicus", "Bombus_steindachneri", "Bombus_subterraneus",
"Bombus_sylvarum", "Bombus_sylvestris", "Bombus_sylvicola", "Bombus_ternarius", "Bombus_terrestris", "Bombus_terricola",
"Bombus_transversalis", "Bombus_ussurensis", "Bombus_vagans_sandersoni", "Bombus_vancouverensis", "Bombus_vandykei",
"Bombus_vestalis", "Bombus_veteranus", "Bombus_volucelloides", "Bombus_vosnesenskii", "Bombus_weisi", "Bombus_wilmattae",
"Bombus_wurflenii", "Bombus_zonatus", "Brachymelecta", "Braunsapis", "Cadeguala", "Calliopsis", "Callomelitta", "Camptopoeum",
"Caupolicana", "Centris", "Ceratina", "Ceylalictus", "Chalepogenus", "Chelostoma", "Coelioxys", "Coleoptera", "Colletes",
"Colletes_cunicularius", "Colletes_hederae", "Colletes_inaequalis", "Corynura", "Dasypoda", "Dasypoda_hirtipes",
"Diadasia", "Dianthidium", "Dieunomia", "Dioxys", "Diphaglossa_gayi", "Diptera", "Dufourea", "Epeoloides", "Epeolus",
"Epicharis", "Ericrocis", "Euaspis", "Eucera", "Eufriesea", "Euglossa", "Euhesma", "Eulaema", "Euryglossa",
"Exaerete", "Exomalopsis", "Exoneura", "Exoneuridia", "Florilegus", "Gaesischia", "Habropoda", "Halictus",
"Halictus_ligatus", "Halictus_rubicundus", "Halictus_scabiosae", "Halictus_tripartitus", "Heriades", "Hesperapis", 
"Holcopasites", "Hoplitis", "Hylaeus", "Hylaeus_leptocephalus", "Hylaeus_modestus", "Hyleoides",
"Icteranthidium", "Lasioglossum", "Leioproctus", "Lepidoptera", "Lipotriches", "Lithurgopsis", "Lithurgus", "Macropis", 
"Macrotera", "Manuelia", "Megachile", "Megachile_ericetorum", "Megachile_latimanus", "Megachile_perihirta",
"Megachile_pugnata", "Megachile_sculpturalis", "Megachile_xylocopoides",
"Megandrena", "Melecta", "Meliponini", "Melissodes", "Melissodes_bimaculatus", "Melissodes_desponsus", "Melissoptila", 
"Melitoma", "Melitta", "Melitturga", "Mellitidia", "Meroglossa", "Mesocheira_bicolor", "Micralictoides", "Nomada", "Nomada_goodeniana",
"Nomada_lathburiana", "Nomia", "Nomioides", "Notanthidium", "Osmia", "Osmia_bicolor", "Osmia_bicornis", "Osmia_caerulescens", 
"Osmia_cornuta", "Osmia_lignaria", "Othinosmia", "Oxaea", "Pachyprosopis", "Palaeorhiza", "Panurginus", "Panurgus",
"Paragapostemon_coelestinus", "Paranthidium", "Paratetrapedia", "Pasites", "Patellapis", "Peponapis", "Perdita", "Protandrena", 
"Protosmia", "Protoxaea", "Pseudapis", "Pseudaugochlora", "Pseudoanthidium", "Pseudopanurgus", "Ptiloglossa", "Ptilothrix", 
"Rhodanthidium", "Rophites", "Ruizantheda", "Scrapter", "Sphecodes", "Stelis", "Svastra", "Syntrichalonia", "Syrphidae",
"Systropha", "Tetralonia", "Tetraloniella", "Thalestria", "Thygater", "Thyreus", "Trachusa", "Trichocolletes", "Triepeolus",
"Wasp", "Xenoglossa", "Xylocopa", "Xylocopa_aestuans", "Xylocopa_augusti", "Xylocopa_caffra", "Xylocopa_californica",
"Xylocopa_flavorufa", "Xylocopa_latipes", "Xylocopa_micans", "Xylocopa_pubescens", "Xylocopa_sonorina",
"Xylocopa_tabaniformis", "Xylocopa_tenuiscapa", "Xylocopa_violacea", "Xylocopa_virginica", "Zacosmia_maculata"
]

classModel = keras.models.load_model('/home/apis/Desktop/pcamData/pcamVids/HTC/BeeMachine_classification_02-13-2024/EfficientNetV2S_300_fp32_2_8_2024.h5')
###############################


IMG_SIZE = 300

directory = "./runs/detect/predict/crops/bee" #directory with cropped images

def analyzeImages(subdir, home, cropHome):
    print(subdir)
    
    unitID, still, cameraID, date = subdir.split('/')[-4:]
    if os.path.exists(os.path.join(home, unitID+'_'+cameraID+'_'+date+'_nobee.txt')) or os.path.exists(os.path.join(home, unitID+'_'+cameraID+'_'+date+'_bees.csv')):
        print('Previously analysed ' + subdir)
        return 0

    df = pd.DataFrame()
    for i in glob.glob(subdir+'/*'):
        #detection model
        time = i.split('_')[-6]
        try:
            results = detectModel(i, save_crop = True, save_dir = home, conf=detectionThres)  # generator of Results objects
        except:
            continue

        for r in results:
            if len(r.boxes.conf) > 0:
                conf =  pd.DataFrame(r.boxes.conf.cpu().numpy(), columns = ['conf'])
                boxes = pd.DataFrame(r.boxes.xywh.cpu().numpy(), columns = ['x', 'y', 'w', 'h'])
                hold = pd.concat([conf, boxes], axis=1)
                hold['originalFile'] = r.path
                hold['short'] = os.path.basename(r.path).split('.')[0]
                suffix = ['']+[str(f+2) for f in range(len(r)-1)]
                hold['filename'] = hold['short'] + suffix + ['.jpg']
                hold['time'] = time
                df = pd.concat([df, hold], axis = 0)
                df.drop('short', axis = 1)

    predictions = {}
    if not os.path.exists(directory):
        f = open(os.path.join(home, unitID+'_'+cameraID+'_'+date+'_nobee.txt'), 'w')
        f.close
        return 0
    
    for i in os.listdir(directory):
        image = os.path.join(directory, i)
        my_image = plt.imread(image)
        my_image_re = resize(my_image, (IMG_SIZE,IMG_SIZE)) 

        probabilities = classModel.predict(np.array( [my_image_re,]))[0,:]

        number_to_class = CATEGORIES
        index = np.argsort(probabilities)
        #save top 3 predictions
        predictions [i] = {
        "class1":number_to_class[index[-1]],
        "class2":number_to_class[index[-2]],
        "class3":number_to_class[index[-3]],
        "prob1":probabilities[index[-1]],
        "prob2":probabilities[index[-2]],
        "prob3":probabilities[index[-3]]
        }

    df2 = pd.DataFrame(predictions)
    df2 = df2.T
    df2['filename'] = df2.index

    
    outDF = df.merge(df2, on='filename') 
    outDF['unitID'] = unitID
    outDF['cameraID'] = cameraID
    outDF['date'] = date
    df.to_csv(os.path.join(home, unitID+'_'+cameraID+'_'+date+'_detection.csv'))
    df2.to_csv(os.path.join(home, unitID+'_'+cameraID+'_'+date+'_classification.csv'))
    outDF.to_csv(os.path.join(home, unitID+'_'+cameraID+'_'+date+'_bees.csv'))
    shutil.copytree(os.path.join(directory), os.path.join(cropHome, unitID+'_'+cameraID+'_'+date))
    shutil.rmtree("./runs")
    return 0

def main(args):
    print(args)
    source = args[1] #source is for each data collection day
    home =  args[2]
    cropHome = args[3]
    if not os.path.isdir(home):
        os.mkdir(home)
    for subdir in glob.glob(source+'/*/*/'+'stills'+'/*/*'):
        analyzeImages(subdir, home, cropHome)
    return 0

if __name__ == "__main__":
    from autopollsStills import *
    main(sys.argv)