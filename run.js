pypyjs.ready().then(function() {
    pypyjs.execfile("connectfour/utils.py");
    pypyjs.execfile("connectfour/games.py");
    
    // Initialize the widget.
    var terminal = $('#terminal').jqconsole('', '>>> ');

    // Hook up output streams to write to the console.
    pypyjs.stdout = pypyjs.stderr = function(data) {
      terminal.Write(data, 'jqconsole-output');
    }
    
    pypyjs.execfile("connectfour/ConnectFour.py");

    // Interact by taking input from the console prompt.
    pypyjs.repl(function(ps1) {

      // The argument is ">>> " or "... " depending on REPL state.
      terminal.SetPromptLabel(ps1);

      // Return a promise if prompting for input asynchronously.
      return new Promise(function(resolve, reject) {
        terminal.Prompt(true, function (input) {
          resolve(input);
        });
      });
    });
});