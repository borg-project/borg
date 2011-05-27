//
// VIEW BASE
//

var newView = function(resources) {
    return {
        onload: function(handler) {
            this.loadHandlers.push(handler);
        },
        loaded: function() {
            for(var i = 0; i < this.loadHandlers.length; i += 1) {
                this.loadHandlers[i].call(this);
            }
        },
        resources: resources,
        loadHandlers: []
    };
};

//
// INDIVIDUAL VIEWS
//

{% include "src.js/views/table.js" %}
{% include "src.js/views/cluster.js" %}

