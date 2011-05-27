//
// GENERAL SUPPORT
//

String.prototype.format = function() {
    var catted = [this];

    return sprintf.apply(null, catted.concat.apply(catted, arguments));
};

var load = function(loadable, callback) {
    var completed = 0;
    var makeHandler = function(request) {
        return function(fetched) {
            loadable[request.name] = fetched;

            completed += 1;

            if(completed == loadable.resources.length) {
                console.log("finished loading " + completed + " resource(s)");

                if(callback !== undefined) {
                    callback();
                }

                loadable.loaded();
            }
        };
    };

    for(var i = 0; i < loadable.resources.length; i += 1) {
        var request = loadable.resources[i];

        d3.json(request.path, makeHandler(request));
    }
};

