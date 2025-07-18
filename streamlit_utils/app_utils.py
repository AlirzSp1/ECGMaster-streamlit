def label_finder(dxs):
    dx_list = dxs[4:].split(',')
    label_json = {
        "426783006": "sinus rhythm",
        "426177001": "sinus bradycardia",
        "164934002": "t wave abnormal",
        "427084000": "sinus tachycardia",
        "164890007": "atrial flutter",
        "39732003": "left axis deviation",
        "164865005": "myocardial infarction",
        "55827005": "Left ventricular hypertrophy",
        "164889003": "atrial fibrillation",
        "55930002": "s t changes",
        "428750005": "nonspecific st t abnormality",
        "164873001": "left ventricular hypertrophy",
        "59931005": "t wave inversion",
        "427393009": "sinus arrhythmia",
        "429622005": "st depression",
        "270492004": "1st degree av block",
        "164951009": "abnormal QRS",
        "59118001": "right bundle branch block",
        "284470004": "premature atrial contraction",
        "164861001": "myocardial ischemia",
        "164930006": "st interval abnormal",
        "445118002": "left anterior fascicular block",
        "164917005": "q wave abnormal",
        "164884008": "ventricular ectopics",
        "111975006": "prolonged qt interval",
        "713426002": "incomplete right bundle branch block",
        "713427006": "complete right bundle branch block",
        "698252002": "nonspecific intraventricular conduction disorder",
        "251146004": "low qrs voltages",
        "10370003": "pacing rhythm",
        "67741000119109": "Left atrial enlargement",
        "164909002": "left bundle branch block",
        "47665007": "right axis deviation",
        "427172004": "premature ventricular contractions",
        "164867002": "old myocardial infarction",
        "425623009": "lateral ischaemia",
        "426761007": "supraventricular tachycardia",
        "425419005": "inferior ischaemia",
        "17338001": "ventricular premature beats",
        "61721007": "Counterclockwise vectorcardiographic",
        "365413008": "R wave",
        "164931005": "st elevation",
        "6374002": "bundle branch block",
        "428417006": "early repolarization",
        "54329005": "anterior myocardial infarction",
        "164947007": "prolonged pr interval",
        "89792004": "right ventricular hypertrophy",
        "713422000": "atrial tachycardia",
        "426434006": "anterior ischemia",
        "233917008": "av block",
        "426627000": "bradycardia",
        "63593006": "supraventricular premature beats",
        "106068003": "Atrial rhythm",
        "251223006": "Tall P wave",
        "733534002": "ECG complete left bundle branch block",
        "251120003": "incomplete left bundle branch block",
        "445211001": "left posterior fascicular block",
        "251199005": "Counterclockwise cardiac rotation",
        "413844008": "chronic myocardial ischemia",
        "74390002": "wolff parkinson white pattern",
        "251200008": "indeterminate cardiac axis",
        "446358003": "right atrial hypertrophy",
        "29320008": "atrioventricular junctional rhythm",
        "164912004": "P wave abnormal",
        "164937009": "u wave abnormal",
        "27885002": "complete heart block",
        "195042002": "2nd degree av block",
        "425856008": "paroxysmal ventricular tachycardia",
        "13640000": "fusion beats",
        "266249003": "ventricular hypertrophy",
        "251205003": "Prolonged P wave",
        "11157007": "ventricular bigeminy",
        "81898007": "ventricular escape rhythm",
        "164896001": "ventricular fibrillation",
        "426995002": "junctional escape",
        "251198002": "Clockwise cardiac rotation",
        "253352002": "left atrial abnormality",
        "251170000": "blocked premature atrial contraction",
        "195126007": "atrial hypertrophy",
        "75532003": "ventricular escape beat",
        "50799005": "Atrioventricular dissociation",
        "57054005": "acute myocardial infarction",
        "251268003": "atrial pacing pattern",
        "446813000": "left atrial hypertrophy",
        "251266004": "ventricular pacing pattern",
        "195080001": "atrial fibrillation and flutter",
        "53741008": "coronary heart disease",
        "251180001": "ventricular trigeminy",
        "67751000119106": "Right atrial enlargement",
        "54016002": "mobitz type i wenckebach atrioventricular block",
        "5609005": "Sinus arrest",
        "426664006": "accelerated junctional rhythm",
        "426648003": "junctional tachycardia",
        "49578007": "shortened pr interval",
        "67198005": "paroxysmal supraventricular tachycardia",
        "233897008": "Re-entrant atrioventricular tachycardia",
        "251182009": "paired ventricular premature complexes",
        "195060002": "ventricular pre excitation",
        "251187003": "Atrial escape complex",
        "233892002": "Ectopic atrial tachycardia",
        "698247007": "cardiac dysrhythmia",
        "61277005": "Accelerated idioventricular rhythm",
        "253339007": "right atrial abnormality",
        "65778007": "sinoatrial block",
        "251164006": "junctional premature complex",
        "251139008": "suspect arm ecg leads reversed",
        "164895002": "ventricular tachycardia",
        "164921003": "r wave abnormal",
        "195101003": "wandering atrial pacemaker",
        "111288001": "ventricular flutter",
        "426183003": "ECG: Mobitz type II atrioventricular block",
        "17366009": "Atrial arrhythmia",
        "84114007": "heart failure",
        "266257000": "transient ischemic attack",
        "368009": "004",
        "251173004": "atrial bigeminy",
        "418818005": "incomplete Brugada syndrome",
        "164942001": "F waves present",
        "77867006": "decreased qt interval",
        "282825002": "paroxysmal atrial fibrillation",
        "413444003": "acute myocardial ischemia",
        "49260003": "idioventricular rhythm",
        "60423000": "sinus node dysfunction",
        "204384007": "congenital incomplete atrioventricular",
        "74615001": "brady tachy syndrome",
        "314208002": "rapid atrial fibrillation",
        "426749004": "chronic atrial fibrillation",
        "251259000": "high t-voltage",
        "251168009": "supraventricular bigeminy",
        "704997005": "inferior ST segment depression",
        "82226007": "diffuse intraventricular block",
        "370365005": "left ventricular strain"
    }
    out_list = []
    for item in dx_list:
        out_list.append(label_json[item])
    return out_list
