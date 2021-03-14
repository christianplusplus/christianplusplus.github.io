pypyjs.ready().then(function() {
    // Initialize the widget.
    var terminal = $('#terminal').jqconsole('', '>>> ');

    // Hook up output streams to write to the console.
    pypyjs.stdout = pypyjs.stderr = function(data) {
      terminal.Write(data, 'jqconsole-output');
    }

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
})