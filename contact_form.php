<?php

if(isset($_POST['submit']))
{
    $name = $_POST['name'];
    $mailFrom = $_POST['email'];
    $company = $_POST['company'];
    $txt = $_POST['subject'];
    
    $mailTo = 'christian.wendlandt.portfolio@outlook.com';
    $headers = 'From: '.$mailFrom;
    $subject = 'portfolio message';
    
    mail($mailTo, $subject, $txt, $headers);
}