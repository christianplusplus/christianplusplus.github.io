function runBot(){
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
            Sk.misceval.callAsync(() =>
                Sk.importMainWithBody('c4blob', false, text, true)
            )
        );
    });
}