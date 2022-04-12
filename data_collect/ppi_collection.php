<?php

include('functions.php');

$export_file = fopen("../data/ppi_intact.csv","w");

for ($i=0; $i<853; $i++){

    $intactEntities = json_decode(file_get_contents("../data/intact/".$i.'.json'));
    foreach($intactEntities->interactions as $interaction){
        $tmp = $interaction->interactor_a->accession . ',' . $interaction->interactor_b->accession . ',' . $interaction->confidence;
        fwrite($export_file, $tmp . "\n");
    }
}

fclose($export_file);

?>
