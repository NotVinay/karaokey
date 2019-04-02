$(document).ready(function() {
    $('#submit').on('click', function(event) {
        $('#loader').modal('show');
        event.stopPropagation();
        event.preventDefault();
        var data = $("#research-form").serialize();
        console.log(data)
        $.ajax({
            type: "POST",
            url: "/submit-research",
            data: data,
            dataType: "json",
            success: function(data) {
                var res = jQuery.parseJSON(data);
                console.log(res)
            },
            error: function() {
            }
        });
    });
});