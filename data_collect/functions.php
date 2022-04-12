<?php

function fetch_data($url){
        #$base = 'https://wwwdev.ebi.ac.uk/crossbar';
        $base = 'https://www.ebi.ac.uk/Tools/crossbar';
        // Create a stream
        $opts = array(
                'http'=>array('timeout' => 5)
        );
        $context = stream_context_create($opts);
        $num_of_tries = 0;
        while($num_of_tries<3){
                # Open the file using the HTTP headers set above
                $file = @file_get_contents($base.$url, false, $context);
                $content = json_decode($file);
                #print_r($content);
                if(!is_object($content)){
                        $error = error_get_last();
                        $f = fopen('data/crossbar_errors.txt', "a");
                        fwrite($f,'Error while fetching data from CROssBAR:'."\n".'Request Url: '.$base.$url."\n".'Time: '.date("Y-m-d H:i:s a")."\nError: ".$error['message']."\n\n");
                        $num_of_tries++;
                        #fwrite($f,$base.$url."\n");
                        # wait for a moment for try again
                        # if we enter this block 3 times, fetching this file will be cancelled
                        sleep(1); # wait 1 second
                        #return false;
                }else
                        break;
        }
        if($num_of_tries===3)
                return false;
        return $content;
}


?>
