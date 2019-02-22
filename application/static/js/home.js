//ROOT = {{ request.script_root|tojson|safe }};
// Variable to store your files


$( document ).ready(function() {
    var file

    $('#music').on('change', function(event) {
        $('#box-error').hide();
        file = event.target.files[0]
        if(file){
            $('#file-name').text(file.name)
        }
    });

    $('#consent').on('change', function(event) {
        $('#consent-error').hide()
    });

    $('#upload').on('click', function() {

        event.stopPropagation(); // stop executing futher events
        event.preventDefault(); // stop current event
        console.log($('#consent').val())
        if(!$('#consent').is(":checked")){
            $('#consent-error').show()
        }
        $('#file-form').prop('disabled', true);
        var data = new FormData();
        data.append('music', file)

        $.ajax({
            url: '/separate',
            type: 'POST',
            data: data,
            cache: false,
            dataType: 'json',
            processData: false, // Don't process the files
            contentType: false, // Set content type to false as jQuery will tell the server its a query string request
            success: function(data, textStatus, jqXHR)
            {
                if(data.token) {
                    // redirect to the results page
                    window.location.replace("/results");
                }
                if(data.error) {
                    // show error
                    $('#file-form').prop('disabled', false);
                    $('#box-error').text(data.error)
                    $('#box-error').addClass("");
                    $('#box-error').show();
                }
            },
            error: function(jqXHR, textStatus, errorThrown)
            {

                $('#file-form').prop('disabled', false);
            }
        });
    });
});
