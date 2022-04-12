<?php
include('functions.php');
/*
https://www.ebi.ac.uk/Tools/crossbar/drugs?limit=1000
identifier
name
canonical_smiles
targets -> accessions
*/

$collected_drugs = array();

for ($i=0; $i<15; $i++)
        if( ($drugEntities = fetch_data('/drugs?limit=1000&page='.$i)) !== false){

                foreach($drugEntities->drugs as $drug){
                        $tmp = array();
                        if($drug->canonical_smiles === NULL)
                                continue;

                        $tmp['id'] = $drug->identifier;
                        $tmp['name'] = $drug->name;
                        $tmp['smiles'] = $drug->canonical_smiles;
                        $tmp['accessions'] = array();

                        # collection drug-target interactions
                        foreach($drug->targets as $target)
                                foreach($target->accessions as $acc){
                                        if($acc !== NULL)
                                                $tmp['accessions'][] = $acc;

                                }
                        $collected_drugs[] = $tmp;
                }
        }

$export_file = fopen("../data/drugs.json","w");
fwrite($export_file, json_encode($collected_drugs));
fclose($export_file);
//echo json_encode($collected_drugs);


?>
