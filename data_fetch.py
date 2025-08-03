from Bio import Entrez
import requests
import time
from xml.etree import ElementTree as ET # For XML parsing
import os
import re
import json

output_dir = "E:/Python/GEN AI/Neuro Research Critics/Data" 
Entrez.email = "abdulwahab41467@gmail.com" # Replace with your email

def search_pubmed_paginated(query, total_to_fetch=1000):
    """Search PubMed with pagination to fetch a specified number of PMIDs."""
    all_pmids = []
    retmax_per_call = 1000 # Max results per ESearch call
    retstart = 0

    print(f"Starting paginated search for up to {total_to_fetch} PMIDs for query: {query}")
    print(f"DEBUG: Actual query string being sent to Entrez:\n---\n{query}\n---") # <--- ADD THIS LINE FOR DEBUGGING
    
    while len(all_pmids) < total_to_fetch:
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query, # This is the 'query' variable we're inspecting
                retmax=str(retmax_per_call),
                retstart=str(retstart),
                retmode="xml",
                tool="MediCritiqueAI",
                email=Entrez.email
            )            
            record = Entrez.read(handle)
            handle.close()

            current_batch_ids = record["IdList"]
            if not current_batch_ids:
                print("No more PMIDs found in current batch. Ending search.")
                break

            all_pmids.extend(current_batch_ids)
            retstart += len(current_batch_ids)
            print(f"  Fetched {len(current_batch_ids)} PMIDs. Total fetched so far: {len(all_pmids)}. Next start index: {retstart}")
            time.sleep(0.5) # Crucial: respect rate limits

            if record.get("Count", "0") and int(record["Count"]) <= len(all_pmids):
                print(f"Reached total count ({record['Count']}) for the query or desired fetch limit.")
                break # All available PMIDs or target fetched

        except Exception as e:
            print(f"Error during PubMed paginated search at retstart {retstart}: {e}")
            break 

    return all_pmids[:total_to_fetch], record.get("Count", "0") 

def get_pmc_ids(pmids):
    """Fetch PMC IDs for a list of PMIDs using ELink in batches."""
    pmc_mapping = {}
    batch_size = 200 
    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i : i + batch_size]
        id_list_str = ",".join(batch_pmids)
        
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=id_list_str,
                retmode="xml",
                tool="MediCritiqueAI",
                email=Entrez.email
            )
            records = Entrez.read(handle)
            handle.close()

            for record in records:
                pubmed_id = record['IdList'][0]
                if 'LinkSetDb' in record:
                    for link_set_db in record['LinkSetDb']:
                        if link_set_db['DbTo'] == 'pmc':
                            for link in link_set_db['Link']:
                                pmc_mapping[pubmed_id] = link['Id']
                                break
                time.sleep(0.1) 
        except Exception as e:
            print(f"Error in elink batch for PMIDs {batch_pmids[0]}-{batch_pmids[-1]}: {e}")
    return pmc_mapping

# Function to download PMC full text XML
def download_pmc_full_text_xml(pmcid):
    """Download full text XML for a given PMC ID."""
    if not pmcid:
        return None
    try:
        handle = Entrez.efetch(
            db="pmc",
            id=pmcid,
            retmode="xml",
            tool="MediCritiqueAI",
            email=Entrez.email
        )
        xml_content = handle.read().decode('utf-8')
        handle.close()
        return xml_content
    except Exception as e:
        print(f"Error downloading PMC XML for PMC{pmcid}: {e}")
        return None

def sanitize_filename(title):
    """core_topic = core_topics()
"""
    # Replace invalid characters with an underscore
    s = re.sub(r'[\\/:*?"<>|]', '_', title)
    s = s.strip()
    # Replace multiple spaces with a single underscore
    s = re.sub(r'\s+', '_', s)
    words = s.split('_')
    words = " ".join(words)
    return words
# --- Main Execution ---

# Define your core topics and their respective queries
def core_topics():
    return {
    "Alzheimer_Disease": {
        "query":  """
    ("Alzheimer Disease"[MeSH] OR "Alzheimer's disease"[tiab])
    AND (therapy[tiab] OR treatment[tiab] OR intervention[tiab] OR drug[tiab] OR "clinical trial"[tiab])
    AND ("2015/01/01"[pdat] : "2025/12/31"[pdat])

""",
        "target_full_text_count": 500,
        "aliases": ["Alzheimer's Disease", "AD", "dementia", "neurodegeneration", "memory loss", "cognitive decline"],
    },
    "Stroke_Management": {
        "query": """
            ("Stroke"[MeSH] OR "Cerebrovascular Accident"[MeSH] OR "Ischemic Stroke"[MeSH] OR "Hemorrhagic Stroke"[MeSH] OR "Stroke"[tiab] OR "Cerebrovascular accident"[tiab])
            AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt])
            AND (
                treatment[tiab] OR therapy[tiab] OR management[tiab] OR intervention[tiab] OR rehabilitation[tiab] OR thrombolysis[tiab] OR thrombectomy[tiab] OR neuroprotection[tiab] OR prevention[tiab] OR outcome[tiab]
            )
            AND ("2015/01/01"[pdat] : "2025/12/31"[pdat])
        """,
        "target_full_text_count": 500, 
        "aliases": ["Cerebrovascular Accident", "CVA", "ischemic stroke", "hemorrhagic stroke", "brain attack", "cerebral infarction"]

    },
       "Epilepsy":{ 
           "query": """
(
    ("Epilepsy"[MeSH] OR "Epilepsies"[MeSH] OR "Drug Resistant Epilepsy"[MeSH] OR "Pharmacoresistant Epilepsy"[MeSH] OR "Refractory Epilepsy"[MeSH] OR "Epilepsy"[tiab] OR "Seizure"[tiab] OR "Epileptic Seizure"[tiab])
    AND ("randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "clinical trial"[pt])
    AND (
        ("drug resistance"[tiab] OR pharmacoresistant[tiab] OR refractory[tiab] OR "novel therapy"[tiab] OR "new drug"[tiab] OR "anti-epileptic drug"[tiab] OR AEDs[tiab] OR "antiepileptic agent"[tiab])
        OR ("vagus nerve stimulation"[tiab] OR VNS[tiab] OR "deep brain stimulation"[tiab] OR DBS[tiab] OR neurostimulation[tiab] OR "epilepsy surgery"[tiab] OR ketogenic[tiab])
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (review[pt] AND NOT "systematic review"[pt] AND NOT "meta-analysis"[pt])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR in vitro[tiab])
)""",
            "target_full_text_count":2000,
        "aliases": ["Seizure Disorder", "Seizures", "Epileptic Syndrome", "Status Epilepticus", "Anti-epileptic Drugs", "AEDs", "Brain Seizures"]

    },
    "Parkinson": {
        "query": """
            (
                ("Parkinson Disease"[MeSH] OR "Parkinson's disease"[tiab] OR "PD"[tiab] OR "Parkinsonian Syndrome"[MeSH] OR "Parkinsonism"[MeSH])
                AND ("journal article"[pt] OR "randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "clinical trial"[pt] OR "review"[pt])
                AND (
                    (motor[tiab] OR "non-motor"[tiab] OR symptoms[tiab] OR management[tiab] OR treatment[tiab] OR therapy[tiab] OR intervention[tiab] OR outcome[tiab] OR prognosis[tiab])
                    OR (dyskinesia[tiab] OR tremor[tiab] OR rigidity[tiab] OR bradykinesia[tiab] OR gait[tiab] OR balance[tiab] OR "cognitive impairment"[tiab] OR hallucination[tiab] OR depression[tiab] OR anxiety[tiab] OR apathy[tiab] OR sleep[tiab] OR pain[tiab] OR autonomic[tiab])
                    OR (DBS[tiab] OR "deep brain stimulation"[tiab] OR levodopa[tiab] OR carbidopa[tiab] OR "MAO-B inhibitor"[tiab] OR dopamine[tiab] OR exercise[tiab] OR rehabilitation[tiab] OR "physical therapy"[tiab] OR "occupational therapy"[tiab])
                )
                AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
                NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "molecular mechanism"[tiab] OR "genetic study"[tiab])
            )
        """.strip().replace('\n', ' ').replace('  ', ' '),
        "target_full_text_count": 2000,
        "aliases": ["Parkinson's Disease", "PD", "Parkinsonism", "Parkinsonian Syndrome", "Motor Disorder", "Movement Disorder"]

    },
    "Diagnostic":{
        "query":  """
(
    ("Diagnosis"[MeSH] OR "Diagnostic Techniques, Neurological"[MeSH] OR "Neuroimaging"[MeSH] OR "Electromyography"[MeSH] OR "Electroencephalography"[MeSH] OR "Magnetic Resonance Imaging"[MeSH] OR "Computed Tomography"[MeSH] OR "Positron-Emission Tomography"[MeSH] OR "Lumbar Puncture"[MeSH] OR "biomarkers"[MeSH] OR "Nerve Conduction"[MeSH] OR "Evoked Potentials"[MeSH])
    AND (
        technique[tiab] OR method[tiab] OR diagnostic[tiab] OR imaging[tiab] OR assessment[tiab] OR "lumbar puncture"[tiab] OR EEG[tiab] OR EMG[tiab] OR NCS[tiab] OR MRI[tiab] OR CT[tiab] OR PET[tiab] OR "cerebrospinal fluid"[tiab] OR "CSF"[tiab] OR "biomarker"[tiab] OR "blood test"[tiab] OR "spinal fluid"[tiab] OR "neuropathology"[tiab] OR "biopsy"[tiab]
    )
    AND ("journal article"[pt] OR review[pt] OR "clinical trial"[pt] OR "case reports"[pt] OR "validation study"[pt] OR "evaluation study"[pt] OR "technical report"[pt] OR "guideline"[pt] OR "consensus development conference"[pt])
    AND ("1995/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (therapy[tiab] OR treatment[tiab] OR intervention[tiab] OR drug[tiab] OR "drug development"[tiab] OR "pharmacology"[tiab] OR "randomized controlled trial"[pt] AND (therapy[tiab] OR treatment[tiab] OR drug[tiab]))
)""".strip().replace('\n', ' ').replace('  ', ' '),
"target_full_text_count":2000,
        "aliases": ["Neurological Diagnosis", "Neuroimaging", "EEG", "EMG", "MRI", "CT scan", "biomarkers", "lumbar puncture", "diagnostic methods"]

    },
    "Neurotransmitter": {
        "query": (
            """
            (
                ("Neurotransmitter Agents"[MeSH] OR "Dopamine"[MeSH] OR "Serotonin"[MeSH] OR "Acetylcholine"[MeSH] OR "GABA"[MeSH] OR "Glutamic Acid"[MeSH] OR "Norepinephrine"[MeSH] OR "Epinephrine"[MeSH] OR "Receptors, Neurotransmitter"[MeSH] OR "Neurochemistry"[MeSH] OR "Synaptic Transmission"[MeSH] OR "Neurotransmitters"[tiab] OR "neurotransmitter systems"[tiab])
                AND (
                    pathway[tiab] OR receptor[tiab] OR synthesis[tiab] OR metabolism[tiab] OR signaling[tiab] OR "synaptic transmission"[tiab] OR "neural circuit"[tiab] OR "brain function"[tiab] OR "neurobiology"[tiab] OR "neuroscience"[tiab] OR function[tiab] OR role[tiab] OR regulation[tiab] OR dysregulation[tiab] OR imbalance[tiab] OR mechanism[tiab]
                )
                AND (
                    disease*[tiab] OR disorder*[tiab] OR pathology[tiab] OR pathophysiology[tiab] OR "neurological disorder"[tiab] OR "mental disorder"[tiab] OR "psychiatric disorder"[tiab]
                    OR "Parkinson Disease"[MeSH] OR "Alzheimer Disease"[MeSH] OR "Depression"[MeSH] OR "Anxiety Disorders"[MeSH] OR "Schizophrenia"[MeSH] OR "Epilepsy"[MeSH] OR "Addiction"[MeSH] OR "Substance-Related Disorders"[MeSH]
                    OR treatment[tiab] OR therapy[tiab] OR drug[tiab] OR pharmacology[tiab]
                )
                AND ("journal article"[pt] OR review[pt] OR "systematic review"[pt] OR "meta-analysis"[pt] OR "clinical trial"[pt] OR "research support, N.I.H., extramural"[pt])
                AND ("1990/01/01"[pdat] : "2025/12/31"[pdat])
                NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "in vivo"[tiab] OR tissue[tiab] OR "cell culture"[tiab])
            )
            """
        ) 
        .replace('\n', ' ') 
        .replace('  ', ' ') 
        .replace('  ', ' ') 
        .strip(),
        "target_full_text_count": 2500,
            "aliases": ["Neurotransmitters", "dopamine", "serotonin", "GABA", "acetylcholine", "neurochemistry", "synaptic transmission", "neurochemicals"]

    },
    "Sclerosis":{
        "query":(
    """
    (
        ("Multiple Sclerosis"[MeSH] OR "MS"[tiab] OR "Demyelinating Diseases"[MeSH] OR "Optic Neuritis"[MeSH])
        AND ("journal article"[pt] OR "randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "clinical trial"[pt] OR "review"[pt])
        AND (
            (diagnosis[tiab] OR treatment[tiab] OR therapy[tiab] OR intervention[tiab] OR management[tiab] OR prognosis[tiab] OR outcome[tiab] OR progression[tiab] OR relapse[tiab] OR remission[tiab])
            OR ("disease-modifying therapy"[tiab] OR DMTs[tiab] OR "immunomodulation"[tiab] OR "immunosuppression"[tiab] OR "natalizumab"[tiab] OR "ocrelizumab"[tiab] OR "fingolimod"[tiab] OR "teriflunomide"[tiab])
        )
        AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
        NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "molecular mechanism"[tiab] OR "genetic study"[tiab])
    )
    """
    
    .replace('\n', ' ')
    .replace('  ', ' ')
    .replace('  ', ' ')
    .strip()
),
   "target_full_text_count": 2000,
    "aliases": ["MS", "Demyelinating Disease", "Optic Neuritis", "immunomodulation", "disease-modifying therapy", "autoimmune neurological disorder"]

    },
"Migraine":{
    "query":("""
(
    ("Migraine"[MeSH] OR "Headache Disorders"[MeSH] OR "Cluster Headache"[MeSH] OR "Tension-Type Headache"[MeSH] OR "Migraine"[tiab] OR "headache"[tiab] OR "cephalalgia"[tiab])
    AND ("journal article"[pt] OR "randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "clinical trial"[pt] OR "review"[pt])
    AND (
        (diagnosis[tiab] OR treatment[tiab] OR therapy[tiab] OR management[tiab] OR prevention[tiab] OR acute[tiab] OR chronic[tiab] OR outcome[tiab])
        OR ("CGRP inhibitor"[tiab] OR "calcitonin gene-related peptide"[tiab] OR "triptan"[tiab] OR "botulinum toxin"[tiab] OR "topiramate"[tiab] OR "onabotulinumtoxinA"[tiab])
    )
    AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "molecular mechanism"[tiab] OR "genetic study"[tiab])
)""".replace('\n', ' ')
    .replace('  ', ' ') 
    .replace('  ', ' ')
    .strip()
),
"target_full_text_count": 2000, 
        "aliases": ["Migraine", "Headache", "Cluster Headache", "Tension Headache", "CGRP inhibitors", "triptans", "cephalalgia"]

},

    "Neurodevelopmental_disorder" : {
        "query": ("""
(
    ("Neurodevelopmental Disorders"[MeSH] OR "Autism Spectrum Disorder"[MeSH] OR "Attention Deficit Disorder with Hyperactivity"[MeSH] OR "Developmental Disabilities"[MeSH] OR "Neurodevelopmental disorder"[tiab] OR "autism"[tiab] OR "ADHD"[tiab] OR "developmental delay"[tiab])
    AND ("journal article"[pt] OR review[pt] OR "systematic review"[pt] OR "meta-analysis"[pt] OR "clinical trial"[pt])
    AND (
        diagnosis[tiab] OR assessment[tiab] OR screening[tiab] OR genetics[tiab] OR etiology[tiab] OR treatment[tiab] OR therapy[tiab] OR intervention[tiab] OR management[tiab] OR outcome[tiab] OR behavior[tiab] OR cognition[tiab]
    )
    AND ("2005/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab])
)""".replace('\n', ' ')
    .replace('  ', ' ') 
    .replace('  ', ' ')
    .strip()),
"target_full_text_count": 2000,  
        "aliases": ["Autism Spectrum Disorder", "ASD", "ADHD", "Attention Deficit Hyperactivity Disorder", "Developmental Disabilities", "Neurodevelopmental Disorders", "Child Neurology"]

    },
    "TBI":{
        "query":("""
(
    ("Traumatic Brain Injury"[MeSH] OR "Concussion"[MeSH] OR "Brain Injuries"[MeSH] OR "TBI"[tiab] OR "concussion"[tiab] OR "head injury"[tiab] OR "post-concussion syndrome"[tiab])
    AND ("journal article"[pt] OR "randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "clinical trial"[pt] OR "review"[pt])
    AND (
        diagnosis[tiab] OR prognosis[tiab] OR outcome[tiab] OR management[tiab] OR treatment[tiab] OR rehabilitation[tiab] OR "long-term effect"[tiab] OR "chronic traumatic encephalopathy"[tiab] OR "CTE"[tiab] OR biomarker[tiab]
    )
    AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "molecular mechanism"[tiab] OR "genetic study"[tiab])
)""".replace('\n', ' ')
    .replace('  ', ' ') 
    .replace('  ', ' ')
    .strip()),
"target_full_text_count": 2000,  
        "aliases": ["Traumatic Brain Injury", "TBI", "Concussion", "Head Injury", "Post-concussion Syndrome", "CTE", "brain trauma"]

    },
    "Amyotrophic_Lateral_Sclerosis":{
        "query":("""
(
    ("Amyotrophic Lateral Sclerosis"[MeSH] OR "Motor Neuron Disease"[MeSH] OR "ALS"[tiab] OR "MND"[tiab] OR "Lou Gehrig's disease"[tiab])
    AND ("journal article"[pt] OR "randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "clinical trial"[pt] OR "review"[pt])
    AND (
        diagnosis[tiab] OR progression[tiab] OR outcome[tiab] OR treatment[tiab] OR therapy[tiab] OR management[tiab] OR biomarker[tiab] OR "disease-modifying"[tiab] OR "riluzole"[tiab] OR "edaravone"[tiab]
    )
    AND ("1995/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "molecular mechanism"[tiab] OR "genetic study"[tiab])
)
""".replace('\n', ' ')
    .replace('  ', ' ') 
    .replace('  ', ' ')
    .strip()),
    "aliases": ["Amyotrophic Lateral Sclerosis", "Motor Neuron Disease", "MND", "Lou Gehrig's Disease", "ALS"],
"target_full_text_count": 2000,  
    },
"Neuroinflammation":{
    "query":("""
    (
        ("Neuroinflammation"[MeSH] OR ("Inflammation"[MeSH] AND "Central Nervous System"[MeSH]))
        OR (neuroinflammation[tiab] OR "brain inflammation"[tiab] OR "glial activation"[tiab] OR "microglia"[tiab] OR "astrocytes"[tiab] OR "cytokines"[tiab] OR "chemokines"[tiab] OR "immune response"[tiab] AND (brain[tiab] OR neurological[tiab] OR CNS[tiab]))
    )
    AND ("journal article"[pt] OR review[pt] OR "systematic review"[pt] OR "meta-analysis"[pt] OR "clinical trial"[pt] OR "research support, N.I.H., extramural"[pt])
    AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (infection[tiab] OR "sepsis"[tiab] OR ("autoimmune disease"[tiab] AND NOT "multiple sclerosis"[tiab]))
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "in vivo"[tiab] OR tissue[tiab] OR "cell culture"[tiab] OR "molecular genetics"[tiab])
    """
    
    .replace('\n', ' ')
    .replace('  ', ' ') 
    .replace('  ', ' ') 
    .strip() 
),
"target_full_text_count": 2000, 
        "aliases": ["Brain Inflammation", "Microglia", "Astrocytes", "Cytokines", "Immune Response in Brain", "Neuroimmune System", "Inflammatory Neuropathy"]

},
    "Sleep_disorder":{
        "query":(
    """
    (
        ("Sleep Disorders"[MeSH] OR "Insomnia"[MeSH] OR "Narcolepsy"[MeSH] OR "Restless Legs Syndrome"[MeSH] OR "Sleep Apnea Syndromes"[MeSH])
        AND (sleep[tiab] OR insomnia[tiab] OR narcolepsy[tiab] OR "restless legs"[tiab] OR "sleep-wake cycle"[tiab] OR "sleep disturbance"[tiab] OR hypersomnia[tiab])
    )
    AND ("journal article"[pt] OR review[pt] OR "clinical trial"[pt] OR "systematic review"[pt] OR "meta-analysis"[pt])
    AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (cardiac[tiab] OR pulmonary[tiab] OR respiratory[tiab] AND NOT "sleep apnea"[tiab])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "molecular mechanism"[tiab] OR "genetic study"[tiab])
    """
    
    .replace('\n', ' ') 
    .replace('  ', ' ') 
    .replace('  ', ' ') 
    .strip()
),
    "target_full_text_count": 2000,  
            "aliases": ["Sleep Disorders", "Insomnia", "Narcolepsy", "Restless Legs Syndrome", "Sleep Apnea", "Sleep Disturbances", "Circadian Rhythm Disorders"]

    },
    "Brain":{
        "query":(
    """
    (
        ("Brain"[MeSH] OR "Central Nervous System"[MeSH] OR "Nervous System"[MeSH] OR "Neurology"[MeSH] OR "Neurosciences"[MeSH])
        OR (brain[tiab] OR neurological[tiab] OR neuroscience[tiab] OR CNS[tiab] OR "nervous system"[tiab] OR "neurobiology"[tiab] OR "brain function"[tiab] OR "human brain"[tiab])
    )
    AND (
        anatomy[tiab] OR physiology[tiab] OR function[tiab] OR structure[tiab] OR overview[tiab] OR review[tiab] OR principles[tiab] OR fundamental[tiab] OR basics[tiab] OR introduction[tiab] OR general[tiab]
    )
    AND ("journal article"[pt] OR review[pt] OR "historical article"[pt] OR "introductory journal article"[pt] OR "encyclopedia"[pt] OR "handbook"[pt])
    AND ("1980/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (disease*[tiab] OR disorder*[tiab] OR clinical[tiab] OR trial[tiab] OR patient*[tiab] OR treatment*[tiab] OR therapy[tiab] OR drug*[tiab] OR diagnosis[tiab] OR management[tiab])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "in vivo"[tiab] OR tissue[tiab] OR "cell culture"[tiab] OR "molecular mechanism"[tiab] OR "genetic study"[tiab] OR "imaging technique"[tiab] OR "biomarker"[tiab])
    """
    .replace('\n', ' ')
    .replace('  ', ' ')
    .replace('  ', ' ')
    .strip()
),  "target_full_text_count": 2000,  
        "aliases": ["Brain Anatomy", "Neuroscience Basics", "Nervous System Function", "Neurobiology Overview", "Brain Physiology", "Human Brain"]

    },
"Social_neurology":{
    "query": (
    """
    (
        ("Stress, Psychological"[MeSH] OR "Emotions"[MeSH] OR "Depression"[MeSH] OR "Depressive Disorder"[MeSH] OR "Anxiety Disorders"[MeSH] OR "Mood Disorders"[MeSH] OR "Mental Disorders"[MeSH] OR "social behavior"[MeSH] OR "social perception"[MeSH] OR "social adjustment"[MeSH])
        OR (stress[tiab] OR "emotional regulation"[tiab] OR "affective disorder"[tiab] OR depression[tiab] OR anxiety[tiab] OR "mood disorder"[tiab] OR "mental health"[tiab] OR "psychiatric illness"[tiab] OR "social cognition"[tiab] OR "social interaction"[tiab] OR "social anxiety"[tiab])
    )
    AND (
        (neurological[tiab] OR neuroscience[tiab] OR neurobiology[tiab] OR "brain function"[tiab] OR "neural circuit"[tiab] OR "CNS"[tiab] OR "central nervous system"[tiab])
        OR ("neurotransmitter"[tiab] OR "HPA axis"[tiab] OR "limbic system"[tiab] OR "brain imaging"[tiab] OR "fMRI"[tiab] OR "EEG"[tiab] OR biomarker[tiab])
    )
    AND (
        treatment[tiab] OR therapy[tiab] OR intervention[tiab] OR management[tiab] OR antidepressant[tiab] OR anxiolytic[tiab] OR psychotherapy[tiab] OR "cognitive behavioral therapy"[tiab] OR "CBT"[tiab] OR "mindfulness"[tiab] OR "deep brain stimulation"[tiab] OR "transcranial magnetic stimulation"[tiab] OR "TMS"[tiab] OR "electroconvulsive therapy"[tiab] OR "ECT"[tiab] OR pharmacology[tiab] OR drug[tiab]
    )
    AND ("journal article"[pt] OR "randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "clinical trial"[pt] OR "review"[pt])
    AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "in vivo"[tiab] OR tissue[tiab] OR "cell culture"[tiab] OR "molecular mechanism"[tiab])
    """
    .replace('\n', ' ')
    .replace('  ', ' ')
    .replace('  ', ' ')
    .strip()
),
 "target_full_text_count": 2000,  
         "aliases": ["Psychological Stress", "Emotional Regulation", "Depression", "Anxiety", "Mood Disorders", "Mental Health", "Social Cognition", "Psychiatric Disorders", "Neuropsychiatry", "Stress-related disorders"]

},
"Nicotine":{
    "query": (
    """
    (
        ("Nicotine"[MeSH] OR "Tobacco Use Disorder"[MeSH] OR "Smoking"[MeSH] OR "Substance-Related Disorders"[MeSH] OR "Drug Addiction"[MeSH] OR "Psychotropic Drugs"[MeSH] OR "Opioid-Related Disorders"[MeSH] OR "Cannabis Use Disorder"[MeSH] OR "Alcoholism"[MeSH])
        OR (nicotine[tiab] OR smoking[tiab] OR tobacco[tiab] OR "vaping"[tiab] OR "e-cigarette"[tiab] OR "substance abuse"[tiab] OR "drug addiction"[tiab] OR "opioid"[tiab] OR "cannabis"[tiab] OR "marijuana"[tiab] OR "alcohol use disorder"[tiab] OR "cocaine"[tiab] OR "methamphetamine"[tiab] OR "addiction"[tiab] OR "withdrawal"[tiab] OR "craving"[tiab] OR "relapse"[tiab])
    )
    AND (
        (neurological[tiab] OR neuroscience[tiab] OR neurobiology[tiab] OR "brain function"[tiab] OR "neural circuit"[tiab] OR "CNS"[tiab] OR "central nervous system"[tiab] OR "neuropharmacology"[tiab])
        OR (dopamine[tiab] OR serotonin[tiab] OR GABA[tiab] OR glutamate[tiab] OR "reward system"[tiab] OR "brain imaging"[tiab] OR "fMRI"[tiab] OR "EEG"[tiab] OR "neurotoxicity"[tiab])
    )
    AND (
        treatment[tiab] OR therapy[tiab] OR intervention[tiab] OR management[tiab] OR "detoxification"[tiab] OR "rehabilitation"[tiab] OR "pharmacotherapy"[tiab] OR "behavioral therapy"[tiab] OR "naltrexone"[tiab] OR "buprenorphine"[tiab] OR "varenicline"[tiab] OR "bupropion"[tiab]
    )
    AND ("journal article"[pt] OR "randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "clinical trial"[pt] OR "review"[pt])
    AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR cell[tiab] OR "in vitro"[tiab] OR "in vivo"[tiab] OR tissue[tiab] OR "cell culture"[tiab] OR "molecular mechanism"[tiab] OR "pharmaceutical chemistry"[MeSH])
    """
    .replace('\n', ' ')
    .replace('  ', ' ')
    .replace('  ', ' ')
    .strip()
),
 "target_full_text_count": 2000, 
         "aliases": ["Substance Abuse", "Drug Addiction", "Nicotine Addiction", "Tobacco Use", "Opioid Use Disorder", "Cannabis Use Disorder", "Alcoholism", "Vaping", "Substance Use Disorder"]
}
}
if __name__ == "__main__":
        
    core_topic= core_topics()
    for topic_name, topic_info in core_topic.items():
        print(f"\n--- Processing Core Topic: {topic_name} ---")
        topic_query = topic_info["query"]
        target_full_text_count = topic_info["target_full_text_count"]
        
        topic_output_dir = os.path.join(output_dir, topic_name)
        os.makedirs(topic_output_dir, exist_ok=True)
        print(f"Output directory for {topic_name}: {topic_output_dir}")

        # Fetch a larger pool of PMIDs, as only a subset will have PMC full text
        pmids_to_fetch_from_search = max(target_full_text_count * 5, 500) # Fetch at least 500, or 5x target
        all_pmids, total_search_count = search_pubmed_paginated(topic_query, total_to_fetch=pmids_to_fetch_from_search) 
        print(f"\nOverall, PubMed search found {total_search_count} articles for '{topic_name}'. Will process {len(all_pmids)} unique PMIDs.")

        # Step 2: Map PMIDs to PMCIDs
        print(f"\nMapping PMIDs to PMCIDs for '{topic_name}'...")
        pmid_to_pmcid = get_pmc_ids(all_pmids)
        pmids_with_pmc = [pmid for pmid, pmcid in pmid_to_pmcid.items() if pmcid]
        print(f"Found {len(pmids_with_pmc)} PMIDs with associated PMCIDs (potential full text) for '{topic_name}'.")

        # Step 3: Download and process full-text XML from PMC
        downloaded_count = 0
        processed_pmids_set = set() # Reset for each topic

        print(f"\nAttempting to download up to {target_full_text_count} full-text XML articles for '{topic_name}'...")
        for pmid in pmids_with_pmc:
            if downloaded_count >= target_full_text_count:
                print(f"Reached target of {target_full_text_count} full-text articles for '{topic_name}'.")
                break
            
            if pmid not in processed_pmids_set:
                pmcid = pmid_to_pmcid[pmid]
                # Attempt to get XML data first, to extract title
                xml_data = download_pmc_full_text_xml(pmcid)
                
                if xml_data:
                    try:
                        root = ET.fromstring(xml_data)
                        article_title_elem = root.find(".//article-meta/title-group/article-title")
                        article_title = article_title_elem.text if article_title_elem is not None else f"No_Title_PMCID_{pmcid}"
                        
                        # --- Extract Main Abstract Text ---
                        main_abstract_parts = []
                        for abstract_elem in root.findall(".//abstract"):
                            abstract_type = abstract_elem.get('abstract-type')
                            if abstract_type is None or abstract_type not in ['graphical', 'web', 'short-communication-abstract']:
                                for p_elem in abstract_elem.findall("p"):
                                    if p_elem.text:
                                        main_abstract_parts.append(p_elem.text.strip())
                        article_abstract = "\n".join(main_abstract_parts) if main_abstract_parts else ""
                        
                        # --- Extract Full Text (from body) ---
                        full_text_parts = []
                        body_elem = root.find(".//body")
                        if body_elem is not None:
                            for paragraph in body_elem.findall(".//p"):
                                if paragraph.text:
                                    full_text_parts.append(paragraph.text.strip())
                        
                        article_full_text = "\n".join(full_text_parts) if full_text_parts else ""

                        content_to_chunk = article_full_text
                        if not content_to_chunk:
                            content_to_chunk = article_abstract
                            print(f"  WARNING: No full text found for PMC{pmcid}. Using abstract as main content for {topic_name}.")
                        
                        if not content_to_chunk: # If still no content, skip or note as very sparse
                            print(f"  WARNING: No content (full text or abstract) found for PMC{pmcid}. Skipping for {topic_name}.")
                            continue # Skip to the next PMID
                            
                        sanitized_title = sanitize_filename(article_title)
                        xml_filename_base = f"{sanitized_title}_PMC{pmcid}"

                        
                        xml_filename = os.path.join(topic_output_dir, f"{xml_filename_base}.xml")
                        
                        # 1. Save the raw XML
                        with open(xml_filename, 'w', encoding='utf-8') as f:
                            f.write(xml_data)

                        try:
                            pub_date_elem = root.find(".//pub-date[@date-type='pub']")
                            year = pub_date_elem.find('year').text if pub_date_elem is not None and pub_date_elem.find('year') is not None else 'N/A'
                        except AttributeError:
                            year = 'N/A'

                        structured_data = {
                            "pmid": pmid,
                            "pmcid": pmcid,
                            "title": article_title,
                            "publication_year": year,
                            "abstract": article_abstract,
                            "full_text": article_full_text,
                            "content_for_embedding": content_to_chunk, # This will be chunked
                            "topic": topic_name # IMPORTANT: Tag the article with its topic
                        }
                        
                        jsonl_filename = os.path.join(topic_output_dir, f"{topic_name.lower()}_articles_data.jsonl")
                        with open(jsonl_filename, 'a', encoding='utf-8') as f:
                            json.dump(structured_data, f, ensure_ascii=False)
                            f.write('\n')
                        print(f"  SUCCESS: Processed PMC{pmcid} for {topic_name}. Saved data to JSONL.")

                        downloaded_count += 1
                        processed_pmids_set.add(pmid)
                        
                    except ET.ParseError as e:
                        print(f"  ERROR parsing XML for PMC{pmcid} for {topic_name}: {e}")
                    except Exception as e:
                        print(f"  An unexpected error occurred processing PMC{pmcid} for {topic_name}: {e}")
                else:
                    print(f"  Skipping PMID: {pmid} as XML data could not be downloaded for {topic_name}.")
                
                time.sleep(0.5)

        print(f"\nFinal count of full-text articles downloaded/processed for '{topic_name}': {downloaded_count}")

    print("\n--- All Core Topics Processed ---")