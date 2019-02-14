//ROOT = {{ request.script_root|tojson|safe }};
// Variable to store your files


$( document ).ready(function() {
    var dragAllowed = function() {
      var tempDiv = document.createElement('div');
      var draggable = ('draggable' in tempDiv) || ('ondragstart' in tempDiv && 'ondrop' in tempDiv)
      return ;
    }();

    var file;

    // Add events
    $("#music").on('change', function( event ){
        file = event.target.files;
    });
});
