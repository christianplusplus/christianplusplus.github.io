function runConsole(){
    var jqconsole = $('#console').jqconsole('', '');

    var output = function(text) {
        jqconsole.Write(text, 'jqconsole-output');
    }

    var input = function(){
        return new Promise(function(resolve, reject) {
            jqconsole.Prompt(
                false,
                function(text){
                    resolve(text);
                }
            );
        });
    }

    Sk.configure({output:output, inputfun:input, __future__:Sk.python3});

    fetch('ConnectFour/c4blob.py').then(response => {
        response.text().then(text =>
            Sk.misceval.callAsync('skulpt',
                () => Sk.importMainWithBody('c4blob', false, text, true)
            )
        );
    });
}

var consoleWorker;

function startWorker(){
    if(false/*typeof(Worker) !== "undefined"*/){
        var jqconsole = $('#console').jqconsole('', '');
        consoleWorker = new Worker("consoleWorker.js");
        consoleWorker.onmessage = function(post){
            console.log(post.data.text);
        }
    }
    else{
        runConsole();
    }
}
