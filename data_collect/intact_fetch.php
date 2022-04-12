<?php

//phpinfo(); die();
for($i=0; $i<853; $i++){

    $base = 'https://www.ebi.ac.uk/Tools/crossbar/intact?limit=1000&page=';
    // Create a stream
    $opts = array(
            'http'=>array('timeout' => 5)
    );
    $context = stream_context_create($opts);
    $num_of_tries = 0;
    $file = "../data/intact/$i.json";
    while($num_of_tries<3){
            # Open the file using the HTTP headers set above
            $raw_data = @file_get_contents($base.$i, false, $context);
            $content = json_decode($raw_data);
            #print_r($content);
            if(!is_object($content)){
                    $error = error_get_last();
                    $f = fopen('fetch_errors.txt', "a");
                    fwrite($f,'Error while fetching data from CROssBAR:'."\n".'Request Url: '.$base.$i."\n".'Time: '.date("Y-m-d H:i:s a")."\nError: ".$error['message']."\n\n");
                    $num_of_tries++;
                    # wait for a moment for try again
                    # if we enter this block 3 times, fetching this file will be cancelled
                    sleep(1); # wait 1 second
            }else{
                file_put_contents($file, $raw_data);
                break;
            }
    }
    if($num_of_tries===3)
            echo 'file ' . $i . ' not downloaded!<br>';
    
}
/*
include('functions.php');
for($i=0; $i<5; $i++){

    $url = '/intact?limit=1000&page='.$i;
    $file = "../data/intact/".$i;

    if( !file_exists( $file )  ){

        //$raw_data = send_request($url);
        $raw_data = fetch_data($url);
        print_r($raw_data);
        //$intact = json_decode( $raw_data );

        /*
        if($intact !== NULL)
            file_put_contents($file, $raw_data);
        else
            echo 'File ' . $i . ' could not downloaded.</br>';
        */
  //  }

//}

?>
