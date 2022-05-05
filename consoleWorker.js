importScripts("imports/skulpt.min.js");
importScripts("imports/skulpt-stdlib.js");

var output = function(text) {
    postMessage({header: 'output', text: text});
}

var inputfun = function(){
    postMessage({header: 'input'});
    return new Promise(function(resolve) {
        onmessage = post => resolve(post.data)
    });
}

Sk.configure({output:output, inputfun:inputfun, __future__:Sk.python3});

fetch('ConnectFour/c4blob.py').then(response => {
    response.text().then(text =>
        Sk.misceval.callAsync('skulpt',
            () => Sk.importMainWithBody('c4blob', false, text, true)
        )
    );
});