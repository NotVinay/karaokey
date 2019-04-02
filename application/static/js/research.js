$(document).ready(function() {
    // hidden at first
    $('#submit-successful').hide()

    // preventing modal from closing when backdrop is clicked
    $('#loader').modal({show: false})

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
            success: function(res, textStatus, jqXHR) {
                console.log(res)
                $('#loader').modal('hide');
                if (res.success) {
                    $('#submit-successful').show()
                    $('#research-form').hide()
                }
            },
            error: function() {
                $('#loader').modal('hide');
            }
        });
    });
});