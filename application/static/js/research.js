$(document).ready(function() {
    // hidden at first
    $('#submit-successful').hide()

    // preventing modal from closing when backdrop is clicked
    $('#loader').modal({show: false})

    $('#submit').on('click', function(event) {
        event.stopPropagation();
        event.preventDefault();
        var i;
        for (i = 1; i < 26; i++){
            console.log('#rating_'+i, $('input[name=rating_'+i+']:checked', '#research-form').val())
            if(!$('input[name=rating_'+i+']:checked', '#research-form').val()){
                alert("Please make sure you have rated all tracks");
                return
                break
            }
        }
        $('#loader').modal('show');
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