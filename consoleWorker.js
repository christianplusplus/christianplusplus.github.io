self.importScripts('imports/skulpt.min.js')
self.importScripts('imports/skulpt-stdlib.js')

var output = function(text) {
    jqconsole.Write(text, 'jqconsole-output');
}

var input = function(input){
    return new Promise(function(resolve) {
        self.onmessage = fuction
    });
}

Sk.configure({output:output, inputfun:input, __future__:Sk.python3});

fetch('ConnectFour/c4blob.py').then(response => {
    response.text().then(text =>
        Sk.importMainWithBody('<stdin>', false, text)
        )
    );
});

self.addEventListener('input', function(post) {
    input(post.data.message)
});