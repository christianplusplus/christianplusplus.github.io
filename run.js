var jqconsole;

var output = function(text) {
    jqconsole.Write(text, 'jqconsole-output');
}

var inputfun = function(){
    return new Promise(function(resolve, reject) {
        jqconsole.Prompt(
            false,
            function(text){
                resolve(text);
            }
        );
    });
}

function startWorker(){
    jqconsole = $('#console').jqconsole('', '');
    if(typeof(Worker) !== "undefined"){
        var consoleWorker = new Worker("consoleWorker.js");
        consoleWorker.onmessage = async function(post){
            console.log(post.data.header);
            switch(post.data.header) {
                case 'output':
                    output(post.data.text);
                    break;
                case 'input':
                    var text = await inputfun();
                    consoleWorker.postMessage(text);
                    break;
            }
        }
    }
    else{
        Sk.configure({output:output, inputfun:inputfun, __future__:Sk.python3});
        fetch('ConnectFour/c4blob.py').then(response => {
            response.text().then(text =>
                Sk.misceval.callAsync('skulpt',
                    () => Sk.importMainWithBody('c4blob', false, text, true)
                )
            );
        });
    }
}
