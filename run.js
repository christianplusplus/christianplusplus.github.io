function runBot(){
    var jqconsole = $('#console').jqconsole('', '');

    var output = async function(text) {
        jqconsole.Write(text, 'jqconsole-output');
    }

    var input = async function(){
        return new Promise(function(resolve, reject) {
            jqconsole.Prompt(
                true,
                function(text){
                    resolve(text);
                }
            );
        });
    }

    Sk.configure({output:output, inputfun:input, __future__:Sk.python3});

    fetch('ConnectFour/c4blob.py').then(response => {
        response.text().then(text => {
            Sk.misceval.asyncToPromise(() =>
                Sk.importMainWithBody('c4blob', false, text, true)
            ).then(
                () => 0
            );
        });
    });
}