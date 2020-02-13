# all datasets from UCR achive: https://www.cs.ucr.edu/~eamonn/time_series_data/
DATASETS = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly',
            'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
            'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X',
            'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
            'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000',
            'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
            'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham',
            'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
            'InsectWingbeatSound', 'ItalyPowerDemand',
            'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT',
            'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup',
            'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain',
            'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
            'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
            'Plane', 'ProximalPhalanxOutlineAgeGroup',
            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
            'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll',
            'SmallKitchenAppliances', 'SonyAIBORobotSurface',
            'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry',
            'SwedishLeaf', 'synthetic_control', 'Symbols', 'ToeSegmentation1', 'ToeSegmentation2',
            'Trace', 'TwoLeadECG', 'Two_Patterns', 'UWaveGestureLibraryAll',
            'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
            'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms',
            'Worms', 'WormsTwoClass', 'yoga']

# dataset used in Multi-Scale paper.
DATASETS_44 = ['50words', 'Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso',
               'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
               'ECGFiveDays', 'FISH', 'FaceAll', 'FaceFour', 'FacesUCR', 'Gun_Point', 'Haptics',
               'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT',
               'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
               'NonInvasiveFatalECG_Thorax2', 'OSULeaf', 'OliveOil', 'SonyAIBORobotSurface',
               'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 'Trace',
               'TwoLeadECG', 'Two_Patterns', 'WordsSynonyms', 'synthetic_control',
               'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z',
               'wafer', 'yoga']


################ Category data set, refer to UCR official website: http://timeseriesclassification.com/dataset.php?train=%3C100&test=&leng=&class=&type=
# Category dataset by train size
DATASETS_CATEGORIZED_BY_TRAINSIZE = {
    'to50': [
        'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'CinC_ECG_torso', 'Coffee',
        'DiatomSizeReduction', 'ECGFiveDays', 'FaceFour', 'Gun_Point', 'MoteStrain', 'OliveOil',
        'ShapeletSim', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'Symbols',
        'ToeSegmentation1','ToeSegmentation2', 'TwoLeadECG'], # 20
    '51to100': [
        'Car', 'ECG200', 'Herring', 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7',
        'MALLAT', 'Meat', 'Trace', 'Wine', 'Worms', 'WormsTwoClass'], # 13
    '101to500': [
        '50words', 'Adiac', 'ChlorineConcentration', 'Computers', 'Cricket_X', 'Cricket_Y',
        'Cricket_Z', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'ECG5000', 'Earthquakes', 'FISH', 'FacesUCR', 'Ham', 'HandOutlines',
        'Haptics', 'InsectWingbeatSound', 'LargeKitchenAppliances', 'MedicalImages',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
        'OSULeaf', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW',
        'RefrigerationDevices', 'ScreenType', 'SmallKitchenAppliances', 'Strawberry', 'SwedishLeaf',
        'WordsSynonyms', 'synthetic_control','yoga'], # 36
    '501to': [
        'ElectricDevices', 'FaceAll', 'FordA', 'FordB', 'NonInvasiveFatalECG_Thorax1',
        'NonInvasiveFatalECG_Thorax2', 'PhalangesOutlinesCorrect', 'ProximalPhalanxOutlineCorrect',
        'ShapesAll', 'StarLightCurves', 'Two_Patterns', 'UWaveGestureLibraryAll',
        'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer'] # 16
}
DATASETS_CATEGORIZED_BY_TESTSIZE = {
    'to300': [
        'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'Coffee', 'Computers', 'ECG200',
        'FISH', 'FaceFour', 'Gun_Point', 'Ham', 'Herring', 'Lighting2', 'Lighting7', 'Meat',
        'OSULeaf', 'OliveOil', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ShapeletSim', 'ToeSegmentation1', 'ToeSegmentation2',
        'Trace', 'Wine', 'Worms', 'WormsTwoClass', 'synthetic_control'], # 29
    '301to1000': [
        '50words', 'Adiac', 'CBF', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
        'ECGFiveDays', 'Earthquakes', 'HandOutlines', 'Haptics', 'InlineSkate',
        'LargeKitchenAppliances', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup',
        'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'PhalangesOutlinesCorrect',
        'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapesAll',
        'SmallKitchenAppliances', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'Strawberry',
        'SwedishLeaf', 'Symbols', 'WordsSynonyms'], # 32
    '1001to': [
        'ChlorineConcentration', 'CinC_ECG_torso', 'ECG5000', 'ElectricDevices', 'FaceAll',
        'FacesUCR', 'FordA', 'FordB', 'InsectWingbeatSound', 'ItalyPowerDemand', 'MALLAT',
        'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'Phoneme',
        'StarLightCurves', 'TwoLeadECG', 'Two_Patterns', 'UWaveGestureLibraryAll',
        'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'yoga']# 24
}
DATASETS_CATEGORIZED_BY_LENGTH = {
    'to300': [
        '50words', 'Adiac', 'ArrowHead', 'CBF', 'ChlorineConcentration', 'Coffee', 'Cricket_X',
        'Cricket_Y', 'Cricket_Z', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll',
        'FacesUCR', 'Gun_Point', 'InsectWingbeatSound', 'ItalyPowerDemand', 'MedicalImages',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
        'MoteStrain', 'PhalangesOutlinesCorrect', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'SonyAIBORobotSurface',
        'SonyAIBORobotSurfaceII', 'Strawberry', 'SwedishLeaf', 'ToeSegmentation1', 'Trace',
        'TwoLeadECG', 'Two_Patterns', 'Wine', 'WordsSynonyms', 'synthetic_control', 'wafer'], # 43
    '301to700': [
        'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'DiatomSizeReduction', 'Earthquakes', 'FISH',
        'FaceFour', 'FordA', 'FordB', 'Ham', 'Herring', 'Lighting2', 'Lighting7', 'Meat', 'OSULeaf',
        'OliveOil', 'ShapeletSim', 'ShapesAll', 'Symbols', 'ToeSegmentation2',
        'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'yoga'], # 25
    '701to': [
        'CinC_ECG_torso', 'Computers', 'HandOutlines', 'Haptics', 'InlineSkate',
        'LargeKitchenAppliances', 'MALLAT', 'NonInvasiveFatalECG_Thorax1',
        'NonInvasiveFatalECG_Thorax2', 'Phoneme', 'RefrigerationDevices', 'ScreenType',
        'SmallKitchenAppliances', 'StarLightCurves', 'UWaveGestureLibraryAll', 'Worms',
        'WormsTwoClass'] # 17
}
DATASETS_CATEGORIZED_BY_CLASSNUMBER = {
    'to10': [
        'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'ChlorineConcentration',
        'CinC_ECG_torso', 'Coffee', 'Computers', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200',
        'ECG5000', 'ECGFiveDays', 'Earthquakes', 'ElectricDevices', 'FISH', 'FaceFour', 'FordA',
        'FordB', 'Gun_Point', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
        'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat',
        'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW', 'MoteStrain', 'OSULeaf', 'OliveOil', 'PhalangesOutlinesCorrect', 'Plane',
        'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'SmallKitchenAppliances',
        'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry',
        'Symbols', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Two_Patterns',
        'UWaveGestureLibraryAll', 'Wine', 'Worms', 'WormsTwoClass', 'synthetic_control',
        'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'yoga'], # 71
    '11to30': [
        'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'FaceAll', 'FacesUCR', 'InsectWingbeatSound',
        'SwedishLeaf', 'WordsSynonyms'], # 8
    '31to': [
        '50words', 'Adiac', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'Phoneme',
        'ShapesAll'] # 6
}

# Category dataset by type,
# Reference:
# UCR official website: http://timeseriesclassification.com/dataset.php?train=%3C100&test=&leng=&class=&type=
DATASETS_CATEGORIZED_BY_TYPES = {
    'device': [
        'Computers','ElectricDevices','LargeKitchenAppliances','RefrigerationDevices','ScreenType',
        'SmallKitchenAppliances'], #6
    'ecg': [
        'CinC_ECG_torso','ECG200', 'ECG5000','ECGFiveDays','NonInvasiveFatalECG_Thorax1',
        'NonInvasiveFatalECG_Thorax2','TwoLeadECG'], #7
    'image': [
        'Adiac','ArrowHead','BeetleFly','BirdChicken','DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW','FaceAll',
        'FaceFour','FacesUCR','50words','FISH','HandOutlines','Herring','MedicalImages',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW','OSULeaf',
        'PhalangesOutlinesCorrect','ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW','ShapesAll','SwedishLeaf', 'Symbols',
        'WordsSynonyms','yoga'], #29
    'motion': [
        'Cricket_X','Cricket_Y', 'Cricket_Z','Gun_Point','Haptics','InlineSkate','ToeSegmentation1',
        'ToeSegmentation2','UWaveGestureLibraryAll', 'uWaveGestureLibrary_X',
        'uWaveGestureLibrary_Y','uWaveGestureLibrary_Z','Worms', 'WormsTwoClass'], #14
    'sensor': [
        'Car','Earthquakes', 'FordA', 'FordB','InsectWingbeatSound', 'ItalyPowerDemand','Lighting2',
        'Lighting7','MoteStrain','Phoneme', 'Plane', 'SonyAIBORobotSurface',
        'SonyAIBORobotSurfaceII', 'StarLightCurves','Trace','wafer'], #16
    'simulated': ['CBF', 'ChlorineConcentration','MALLAT','ShapeletSim','synthetic_control',
                  'Two_Patterns'], #6
    'spectrum': ['Beef','Coffee','Ham','Meat','OliveOil','Strawberry','Wine'] #7
}


















