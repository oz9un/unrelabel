// unrelabel/static/js/charts.js
// D3.js v7 chart functions: v2 with zoom, tooltips, yellow ring highlights
(function() {
  'use strict';

  var COLORS = {
    bg: '#000000',
    surface: '#111111',
    border: '#222222',
    red: '#ff0000',
    text: '#ffffff',
    muted: '#aaaaaa',
    tertiary: '#555555',
    highlight: '#f0c000',
  };

  var CLASS_COLORS = ['#ff0000', '#00ff00', '#ffaa00', '#00f0ff', '#ff66ff', '#ff8800', '#66ff66', '#8888ff'];

  // ---- Scatter Plot ----
  window.drawScatter = function(containerId, data, options) {
    var opts = options || {};
    var container = document.getElementById(containerId);
    if (!container) return;

    if (data.strategy === 'stats') {
      if (data.stats) {
        var classNames = Object.keys(data.stats);
        var means = classNames.map(function(c) {
          var arr = data.stats[c];
          return arr.reduce(function(a, b) { return a + b; }, 0) / arr.length;
        });
        drawBarChart(containerId, means, classNames, 'Mean Feature Value', false);
      } else {
        container.innerHTML = '<p style="color:' + COLORS.muted + '">High-dimensional data: stats unavailable.</p>';
      }
      return;
    }
    container.innerHTML = '';

    var pointRadius = opts.large ? 6 : 5;
    var highlightRadius = opts.large ? 11 : 10;
    var margin = { top: 10, right: 20, bottom: 40, left: 50 };
    var width = (container.clientWidth || 400) - margin.left - margin.right;
    var height = (opts.height || 320) - margin.top - margin.bottom;

    var svg = d3.select('#' + containerId)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var coords = data.coords;
    var xExt = d3.extent(coords, function(d) { return d[0]; });
    var yExt = d3.extent(coords, function(d) { return d[1]; });

    var x = d3.scaleLinear().domain(xExt).range([0, width]).nice();
    var y = d3.scaleLinear().domain(yExt).range([height, 0]).nice();

    // X axis
    var xAxisLabel = 'Feature 1';
    var yAxisLabel = 'Feature 2';
    if (data.explained_variance && data.explained_variance.length >= 2) {
      xAxisLabel = 'PC1 (' + (data.explained_variance[0] * 100).toFixed(1) + '%)';
      yAxisLabel = 'PC2 (' + (data.explained_variance[1] * 100).toFixed(1) + '%)';
    }

    svg.append('g').attr('transform', 'translate(0,' + height + ')')
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text').style('fill', COLORS.muted).style('font-size', '12px').style('font-family', 'Anonymous Pro, monospace');
    svg.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text').style('fill', COLORS.muted).style('font-size', '12px').style('font-family', 'Anonymous Pro, monospace');

    svg.selectAll('.domain, .tick line').style('stroke', COLORS.border);

    // Axis titles
    svg.append('text')
      .attr('x', width / 2).attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .style('fill', COLORS.muted).style('font-size', '13px')
      .style('font-family', 'Anonymous Pro, monospace')
      .text(xAxisLabel);
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', COLORS.muted).style('font-size', '13px')
      .style('font-family', 'Anonymous Pro, monospace')
      .text(yAxisLabel);

    // Points
    var highlightSet = data.highlight_indices ? new Set(data.highlight_indices) : null;
    var pointData = coords.map(function(c, i) {
      return { x: c[0], y: c[1], label: data.labels[i], idx: i };
    });

    svg.selectAll('circle.point')
      .data(pointData)
      .enter()
      .append('circle')
      .attr('class', 'point')
      .attr('cx', function(d) { return x(d.x); })
      .attr('cy', function(d) { return y(d.y); })
      .attr('r', pointRadius)
      .attr('fill', function(d) { return CLASS_COLORS[d.label % CLASS_COLORS.length]; })
      .attr('opacity', function(d) { return (highlightSet && !highlightSet.has(d.idx)) ? 0.25 : 0.7; });

    // Yellow ring highlights
    if (highlightSet) {
      var hlData = pointData.filter(function(d) { return highlightSet.has(d.idx); });
      svg.selectAll('circle.highlight-ring')
        .data(hlData)
        .enter()
        .append('circle')
        .attr('class', 'highlight-ring')
        .attr('cx', function(d) { return x(d.x); })
        .attr('cy', function(d) { return y(d.y); })
        .attr('r', highlightRadius)
        .attr('fill', 'none')
        .attr('stroke', COLORS.highlight)
        .attr('stroke-width', 2.5);
    }

    // Tooltip
    var tooltip = d3.select('body').append('div')
      .attr('class', 'scatter-tooltip')
      .style('display', 'none');

    svg.selectAll('circle.point')
      .on('mouseover', function(event, d) {
        var className = data.class_names[d.label] || ('Class ' + d.label);
        tooltip.html(
          'Sample #' + d.idx + '<br>' +
          '<span class="tt-class">' + className + '</span><br>' +
          'x: ' + d.x.toFixed(3) + ', y: ' + d.y.toFixed(3)
        );
        tooltip.style('display', 'block')
          .style('left', (event.pageX + 12) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mousemove', function(event) {
        tooltip.style('left', (event.pageX + 12) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function() {
        tooltip.style('display', 'none');
      });

    // Legend
    var legend = svg.append('g').attr('transform', 'translate(' + (width - 90) + ', 5)');
    data.class_names.forEach(function(name, i) {
      var g = legend.append('g').attr('transform', 'translate(0,' + (i * 18) + ')');
      g.append('rect').attr('width', 10).attr('height', 10).attr('fill', CLASS_COLORS[i % CLASS_COLORS.length]);
      g.append('text').attr('x', 14).attr('y', 10).text(name)
        .style('fill', COLORS.muted).style('font-size', '12px').style('font-family', 'Anonymous Pro, monospace');
    });

    // Highlight legend
    if (highlightSet && highlightSet.size > 0) {
      var hlLegend = legend.append('g').attr('transform', 'translate(0,' + (data.class_names.length * 18 + 4) + ')');
      hlLegend.append('circle').attr('cx', 5).attr('cy', 5).attr('r', 5).attr('fill', COLORS.red).attr('opacity', 0.7);
      hlLegend.append('circle').attr('cx', 5).attr('cy', 5).attr('r', 9).attr('fill', 'none').attr('stroke', COLORS.highlight).attr('stroke-width', 2);
      hlLegend.append('text').attr('x', 14).attr('y', 10).text('poisoned')
        .style('fill', COLORS.highlight).style('font-size', '12px').style('font-family', 'Anonymous Pro, monospace');
    }

    // Re-add zoom hint after rendering
    if (!opts.large) {
      var hint = document.createElement('span');
      hint.className = 'zoom-hint';
      hint.textContent = 'CLICK TO ZOOM';
      container.appendChild(hint);
    }

    // Zoom click: attach to container, not SVG (so zoom hint works too)
    if (typeof openZoom === 'function' && !opts.large) {
      container.onclick = function() {
        openZoom(function(targetId) {
          drawScatter(targetId, data, { large: true, height: Math.min(window.innerHeight * 0.7, 600) });
        });
      };
    }
  };

  // ---- Confusion Matrix ----
  window.drawConfusionMatrix = function(containerId, cm, classNames, title, animate, options) {
    var opts = options || {};
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    var n = cm.length;
    var cellSize = opts.large ? 64 : 52;
    var fontSize = opts.large ? 18 : 16;
    var labelSize = opts.large ? 13 : 12;
    // Wider margins for rotated column labels and row labels + TRUE axis
    var margin = { top: 60, right: 10, bottom: 50, left: 80 };
    var size = n * cellSize;

    var svg = d3.select('#' + containerId)
      .append('svg')
      .attr('width', size + margin.left + margin.right)
      .attr('height', size + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // Title
    svg.append('text')
      .attr('x', size / 2).attr('y', -20)
      .attr('text-anchor', 'middle')
      .style('fill', COLORS.text).style('font-size', '14px')
      .style('font-family', 'Archivo Black, sans-serif')
      .text(title);

    // Max off-diagonal
    var maxOff = 0;
    for (var i = 0; i < n; i++)
      for (var j = 0; j < n; j++)
        if (i !== j) maxOff = Math.max(maxOff, cm[i][j]);
    if (maxOff === 0) maxOff = 1;

    // Cells
    var cells = [];
    for (var ii = 0; ii < n; ii++)
      for (var jj = 0; jj < n; jj++)
        cells.push({ row: ii, col: jj, val: cm[ii][jj] });

    svg.selectAll('rect.cell')
      .data(cells)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', function(d) { return d.col * cellSize; })
      .attr('y', function(d) { return d.row * cellSize; })
      .attr('width', cellSize - 2)
      .attr('height', cellSize - 2)
      .attr('fill', COLORS.surface)
      .attr('stroke', function(d) { return d.row === d.col ? '#333333' : 'none'; })
      .attr('stroke-width', function(d) { return d.row === d.col ? 1 : 0; })
      .transition()
      .delay(function(d, i) { return animate ? i * 50 : 0; })
      .duration(animate ? 200 : 0)
      .attr('fill', function(d) {
        if (d.row === d.col) return '#1a1a1a';
        if (d.val === 0) return COLORS.surface;
        var alpha = 0.2 + 0.8 * (d.val / maxOff);
        return d3.interpolateRgb(COLORS.surface, COLORS.red)(alpha);
      });

    // Cell text
    svg.selectAll('text.cell-text')
      .data(cells)
      .enter()
      .append('text')
      .attr('class', 'cell-text')
      .attr('x', function(d) { return d.col * cellSize + cellSize / 2; })
      .attr('y', function(d) { return d.row * cellSize + cellSize / 2; })
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .style('font-size', fontSize + 'px')
      .style('font-family', 'Anonymous Pro, monospace')
      .style('font-weight', function(d) { return d.row === d.col ? '700' : '400'; })
      .style('fill', function(d) {
        if (d.row === d.col) return COLORS.text;
        if (d.val === 0) return COLORS.tertiary;
        // White text on intense red cells for readability
        var intensity = d.val / maxOff;
        return intensity > 0.4 ? COLORS.text : '#ff4444';
      })
      .style('opacity', 0)
      .transition()
      .delay(function(d, i) { return animate ? i * 50 : 0; })
      .duration(animate ? 200 : 0)
      .style('opacity', 1)
      .text(function(d) { return d.val; });

    // Column labels (top): rotated to avoid overlap
    classNames.forEach(function(name, i) {
      var displayName = opts.large ? name : (name.length > 8 ? name.slice(0, 7) + '.' : name);
      var xPos = i * cellSize + cellSize / 2;
      svg.append('text')
        .attr('transform', 'translate(' + xPos + ',-8) rotate(-35)')
        .attr('text-anchor', 'end')
        .style('fill', COLORS.muted).style('font-size', labelSize + 'px')
        .style('font-family', 'Anonymous Pro, monospace')
        .text(displayName);
    });

    // Row labels (left)
    classNames.forEach(function(name, i) {
      var displayName = opts.large ? name : (name.length > 8 ? name.slice(0, 7) + '.' : name);
      svg.append('text')
        .attr('x', -10)
        .attr('y', i * cellSize + cellSize / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'central')
        .style('fill', COLORS.muted).style('font-size', labelSize + 'px')
        .style('font-family', 'Anonymous Pro, monospace')
        .text(displayName);
    });

    // Axis titles
    svg.append('text').attr('x', size / 2).attr('y', size + 35)
      .attr('text-anchor', 'middle')
      .style('fill', COLORS.muted).style('font-size', '13px')
      .style('font-family', 'Anonymous Pro, monospace')
      .text('PREDICTED');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -size / 2).attr('y', -68)
      .attr('text-anchor', 'middle')
      .style('fill', COLORS.muted).style('font-size', '13px')
      .style('font-family', 'Anonymous Pro, monospace')
      .text('TRUE');
  };

  // ---- Bar Chart ----
  window.drawBarChart = function(containerId, values, labels, yLabel, animate) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    var margin = { top: 20, right: 10, bottom: 35, left: 55 };
    var width = (container.clientWidth || 400) - margin.left - margin.right;
    var height = 200 - margin.top - margin.bottom;

    var svg = d3.select('#' + containerId)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var x = d3.scaleBand().domain(labels).range([0, width]).padding(0.3);
    var yMax = d3.max(values) * 1.1 || 1;
    var yScale = d3.scaleLinear().domain([0, yMax]).range([height, 0]).nice();

    svg.append('g').attr('transform', 'translate(0,' + height + ')')
      .call(d3.axisBottom(x))
      .selectAll('text').style('fill', COLORS.muted).style('font-size', '13px').style('font-family', 'Anonymous Pro, monospace');
    svg.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text').style('fill', COLORS.muted).style('font-size', '12px').style('font-family', 'Anonymous Pro, monospace');

    svg.selectAll('.domain, .tick line').style('stroke', COLORS.border);

    // Bars
    svg.selectAll('rect.bar')
      .data(values)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', function(d, i) { return x(labels[i]); })
      .attr('width', x.bandwidth())
      .attr('y', height)
      .attr('height', 0)
      .attr('fill', COLORS.red)
      .attr('opacity', 0.8)
      .transition()
      .delay(function(d, i) { return animate ? i * 100 : 0; })
      .duration(animate ? 400 : 0)
      .attr('y', function(d) { return yScale(d); })
      .attr('height', function(d) { return height - yScale(d); });

    // Value labels on bars
    svg.selectAll('text.bar-label')
      .data(values)
      .enter()
      .append('text')
      .attr('class', 'bar-label')
      .attr('x', function(d, i) { return x(labels[i]) + x.bandwidth() / 2; })
      .attr('y', function(d) { return yScale(d) - 6; })
      .attr('text-anchor', 'middle')
      .style('fill', COLORS.text).style('font-size', '13px')
      .style('font-family', 'Anonymous Pro, monospace')
      .style('font-weight', '700')
      .text(function(d) { return typeof d === 'number' ? (d * 100).toFixed(1) + '%' : d; });

    // Y-axis label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2).attr('y', -40)
      .attr('text-anchor', 'middle')
      .style('fill', COLORS.muted).style('font-size', '13px')
      .style('font-family', 'Anonymous Pro, monospace')
      .text(yLabel || '');

    // Zoom click: attach to parent .chart-container if present (so zoom hint is clickable too)
    if (typeof openZoom === 'function') {
      var clickTarget = container.closest('.chart-container') || container;
      clickTarget.onclick = function() {
        openZoom(function(targetId) {
          drawBarChart(targetId, values, labels, yLabel, false);
        });
      };
    }
  };

})();
