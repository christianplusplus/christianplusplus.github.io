importScripts("imports/skulpt.min.js");
importScripts("imports/skulpt-stdlib.js");

var output = function(text) {
    postMessage(text);
}

var input = function(){
    postMessage('input');
    return new Promise(function(resolve) {
        onmessage = post => resolve(post.data.text)
    });
}

Sk.configure({output:output, inputfun:input, __future__:Sk.python3});

fetch('ConnectFour/c4blob.py').then(response => {
    response.text().then(text =>
        Sk.importMainWithBody('<stdin>', false, text)
    );
});