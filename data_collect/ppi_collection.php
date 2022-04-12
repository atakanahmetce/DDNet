<?php

include('functions.php');

$export_file = fopen("../data/ppi_intact.csv","w");

$ppi_data = array();

for ($i=0; $i<853; $i++)
    if( ($intactEntities = fetch_data('/intact?limit=1000&page='.$i)) !== false){

        foreach($intactEntities->interactions as $interaction){
            $tmp = $interaction->interactor_a->accession . ',' . $interaction->interactor_b->accession . ',' . $interaction->confidence;
            if ( !in_array($tmp, $ppi_data) )
                $ppi_data[] = $tmp;
        }

    }

foreach($ppi_data as $ppi)
    fwrite($export_file, $ppi . "\n");

//print_r($ppi_data);

fclose($export_file);

?>
