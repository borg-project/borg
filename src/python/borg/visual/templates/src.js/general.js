//
// GENERAL SUPPORT
//

String.prototype.format = function() {
    var catted = [this];

    return sprintf.apply(null, catted.concat.apply(catted, arguments));
};

var load = function(loadable) {
    var remaining = loadable.resourcesRequested.length;
    var $loadable = $(loadable);

    $loadable.bind("resource-loaded", function(event, request, resource) {
        loadable.resources[request.name] = resource;

        remaining -= 1;

        if(remaining === 0) {
            console.log("finished loading %s resource(s)".format(loadable.resourcesRequested.length));

            $loadable.trigger("resources-loaded");
        }
    });

    loadable.resourcesRequested.forEach(function(request) {
        d3.json(request.path, function(resource) {
            $loadable.trigger("resource-loaded", [request, resource]);
        });
    });
};

