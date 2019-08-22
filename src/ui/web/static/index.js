$(function() {
  $.getJSON("./classifiers").done(function(result) {
    var generic = result.general
      .map(function(model) {
        return (
          '<option value="class=general&model=' +
          model +
          '">Generic ' +
          model.replace("_", " ").toUpperCase() +
          "</option>"
        );
      })
      .join("");

    var specific = result.specific
      .map(function(model) {
        return (
          '<option value="class=specific&model=' +
          model +
          '">Specific ' +
          model.replace("_", " ").toUpperCase() +
          "</option>"
        );
      })
      .join("");

    $("#dropdown").html(generic + specific);
  });
  $(".file-input").on("change", function(e) {
    e.preventDefault();
    var data = new FormData();
    var file = $(".file-input")[0].files[0];
    $("#classifier-result a#save-image").attr(
      "download",
      "classified_" + file.name
    );

    data.append("file", file);
    $("#upload-icon")
      .addClass("fa-cog fa-spin")
      .removeClass("fa-upload");

    $(".file").removeClass("is-primary");
    $(".file-cta span.file-label").text("Classifying image...");
    $(".progress").removeClass("is-hidden");

    $("#file-fieldset").attr("disabled", true);

    var queryString = $("#dropdown").val();
    $.ajax("/classify?" + queryString, {
      processData: false,
      contentType: false,
      data: data,
      type: "POST",
      beforeSend: function(xhr) {
        xhr.overrideMimeType("text/plain; charset=x-user-defined");
      },
      success: loaded
    }).fail(function() {
      alert("No planktons were found on your image!");
      restore();
    });
  });

  function loaded(result, textStatus, jqXHR) {
    if (result.length < 1) {
      alert("No planktons were found on your image!");
    } else {
      var binary = "";
      var responseText = jqXHR.responseText;
      var responseTextLen = responseText.length;

      for (i = 0; i < responseTextLen; i++) {
        binary += String.fromCharCode(responseText.charCodeAt(i) & 255);
      }
      $("#classifier-result img").attr(
        "src",
        "data:image/png;base64," + btoa(binary)
      );
      $("#classifier-result a#save-image").attr(
        "href",
        "data:image/png;base64," + btoa(binary)
      );
      $("#classifier-result").removeClass("is-hidden");
    }
    restore();
  }

  function restore() {
    $("#upload-icon")
      .removeClass("fa-cog fa-spin")
      .addClass("fa-upload");
    $(".file").addClass("is-primary");
    $(".file-cta span.file-label").text("Choose another image");
    $("#file-fieldset").attr("disabled", false);
    $(".progress").addClass("is-hidden");
  }
});
