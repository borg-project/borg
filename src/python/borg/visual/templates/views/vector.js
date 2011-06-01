    view.onload(function() {
        // compute membership vectors
        // XXX don't hard-code cluster count
        //var vectors = [];
        var clusters = 15;
        //var angle = 2 * Math.PI / clusters;

        //for(var i = 0; i < clusters; i += 1) {
            //vectors[i] = {x: Math.sin(angle * i) / 2, y: Math.cos(angle * i) / 2};
        //}

        // compute instance node positions
        var nodes = [];
        var xDomain = [null, null];
        var yDomain = [null, null];

        for(var i = 0; i < view.instances.length; i += 1) {
            //var node = {x: 0.5, y: 0.5, name: view.instances[i], dominant: 0};

            //for(var j = 0; j < clusters; j += 1) {
                //var m = view.membership[i][j];

                //if(m > view.membership[i][node.dominant]) {
                    //node.dominant = j;
                //}

                //node.x += m * vectors[j].x;
                //node.y += m * vectors[j].y;
            //}

            var node = {
                x: this.projection[i][0],
                y: this.projection[i][1],
                name: this.instances[i],
                dominant: 0
            };

            for(var j = 0; j < clusters; j += 1) {
                if(this.membership[i][j] > this.membership[i][node.dominant]) {
                    node.dominant = j;
                }
            }

            if(xDomain[0] === null || node.x < xDomain[0]) { xDomain[0] = node.x; }
            if(xDomain[1] === null || node.x > xDomain[1]) { xDomain[1] = node.x; }
            if(yDomain[0] === null || node.y < yDomain[0]) { yDomain[0] = node.y; }
            if(yDomain[1] === null || node.y > yDomain[1]) { yDomain[1] = node.y; }

            nodes[i] = node;
        }

        // prepare node shapes
        var $clusterView = $('<svg id="cluster-view"></svg>').appendTo("#display");
        // XXX
        //var defs = d3.select("#cluster-view").append("svg:defs");
        //var defs = d3.select("#cluster-view");

        //defs.append("svg:circle").attr("id", "foo-rect");

        // add cluster vectors
        var xScale = d3.scale.linear().domain(xDomain).range([16, $clusterView.innerWidth() - 16]);
        var yScale = d3.scale.linear().domain(yDomain).range([16, $clusterView.innerHeight() - 16]);
        var colors = d3.scale.category20();

        //d3.select("#cluster-view")
            //.selectAll(".cluster-arrow")
            //.data(vectors)
            //.enter()
            //.append("svg:line")
            //.attr("class", "cluster-arrow")
            //.attr("x1", function(d) { return xScale(0.5); })
            //.attr("y1", function(d) { return yScale(0.5); })
            //.attr("x2", function(d) { return xScale(0.5 + d.x); })
            //.attr("y2", function(d) { return yScale(0.5 + d.y); })
            //.style("stroke", function(d, i) { return colors(i / 16.0); });

        // add instance nodes
        d3.select("#cluster-view")
            .selectAll(".instance-point")
            .data(nodes)
            .enter()
            .append("svg:circle")
            //.attr("xlink:href", "#foo-rect")
            .attr("class", "instance-point")
            .attr("r", 8)
            .attr("cx", function(d) { return xScale(d.x); })
            .attr("cy", function(d) { return yScale(d.y); })
            .style("fill", function(d) { return colors(d.dominant / clusters); });
    });

