//
// LOADING SCREEN

bv.loading = {};

bv.loading.create = function(view) {
    var loading = Object.create(bv.loading);
    var markup = '<div id="bv-loading-dialog" title="Loading Resources"><p id="bv-loading-p">Starting to load resources.</p><div id="bv-loading-progress"></div></div>';

    $(markup)
        .appendTo("body")
        .dialog({
            resizable: false,
            modal: true,
            draggable: false,
            closeOnEscape: false,
            minHeight: 100
        })
        .find("#bv-loading-progress")
        .progressbar();

    var count = 0;

    $(view).bind("resource-loaded", function() {
        count += 1;

        var total = view.resourcesRequested.length;
        var value = count / total;
        var description = "Transfer complete for %s of %s resources.".format(count, total);

        $("#bv-loading-p").text(description);
        $("#bv-loading-progress").progressbar("option", "value", value * 100);
    });
    $(view).bind("resources-loaded", function() {
        $("#bv-loading-dialog").remove();
    });
};

