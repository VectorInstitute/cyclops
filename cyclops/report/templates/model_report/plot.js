// Javascript code for plots in the model card template
const MAX_SIZE = 20;
let maxSampleSize = 0;

// Define a function to update the plot based on selected filters
function updatePlot() {
  const inputs = document.querySelectorAll('#slice-selection input[type="radio"]:checked');
  var plot_selection = document.querySelectorAll('#plot-selection input[type="radio"]');
  var plot_selected = document.querySelectorAll('#plot-selection input[type="radio"]:checked')[0];
  // get number from value in plot_selected "Plot 1" -> 1
  var label_selection = document.querySelectorAll('#plot-selection label');
  var label_slice_selection = document.querySelectorAll('#slice-selection label');

  var mean_std_plot_selection = document.querySelectorAll('#mean-std-selection input[type="checkbox"]');
  var mean_plot_selection = mean_std_plot_selection[0];
  var std_plot_selection = mean_std_plot_selection[1];

  var mean_std_label_selection = document.querySelectorAll('#mean-std-selection label');
  var mean_label_selection = mean_std_label_selection[0];
  var std_label_selection = mean_std_label_selection[1];

  // get all inputs values from div class radio-buttons
  // get name of inputs
  var inputs_name = [];
  var inputs_value = [];
  for (let i = 0; i < inputs.length; i++) {
      inputs_name.push(inputs[i].name);
      inputs_value.push(inputs[i].value);
  }

  var plot_number = parseInt(plot_selected.value.split(" ")[1]-1);
  var selection = [];
  for (let i = 0; i < inputs_value.length; i++) {
      selection.push(inputs_name[i] + ":" + inputs_value[i]);
  }
  selection.sort();
  selections[plot_number] = selection;

  // if plot_selected is "+" then add new radio button to plot_selection called "Plot N" where last plot is N-1 but keep "+" at end and set new radio button to checked for second last element
  if (plot_selected.value === "+") {
      // if 10 plots already exist, don't add new plot and gray out "+"
      if (plot_selection.length === 13) {
      plot_selected.checked = false;
      label_selection[-1].style.color = "gray";
      return;
      }
      var new_plot = document.createElement("input");
      new_plot.type = "radio";
      new_plot.id = "Plot " + (plot_selection.length);
      new_plot.name = "plot";
      new_plot.value = "Plot " + (plot_selection.length);
      new_plot.checked = true;
      var new_label = document.createElement("label");
      new_label.htmlFor = "Plot " + (plot_selection.length);
      new_label.innerHTML = "Plot " + (plot_selection.length);

      // Parse plot_color to get r, g, b values
      var plot_color = plot_colors[plot_selection.length]
      const [r, g, b] = plot_color.match(/\d+/g);
      const rgbaColor = `rgba(${r}, ${g}, ${b}, 0.2)`;
      // set background color of new radio button to plot_color
      new_label.style.backgroundColor = rgbaColor;
      new_label.style.border = "2px solid " + plot_color;
      new_label.style.color = plot_color;

      // insert new radio button and label before "+" radio button and after last radio button
      plot_selected.insertAdjacentElement("beforebegin", new_plot);
      plot_selected.insertAdjacentElement("beforebegin", new_label);
      // Add event listener to new radio button
      new_plot.addEventListener('change', updatePlot);

      // set plot_selected to new plot
      plot_selected = new_plot

      for (let i = 0; i < label_selection.length-1; i++) {
      plot_selection[i].checked = false;
      label_selection[i].style.backgroundColor = "#ffffff";
      label_selection[i].style.border = "2px solid #DADCE0";
      label_selection[i].style.color = "#000000";
      }
      if (mean_plot_selection.checked) {
        mean_label_selection.style.backgroundColor = rgbaColor;
        mean_label_selection.style.border = "2px solid " + plot_color;
        mean_label_selection.style.color = plot_color;
        }
        else {
        mean_label_selection.style.backgroundColor = "#ffffff";
        mean_label_selection.style.border = "2px solid #DADCE0";
        mean_label_selection.style.color = "#000000";
        }
      if (std_plot_selection.checked) {
        std_label_selection.style.backgroundColor = rgbaColor;
        std_label_selection.style.border = "2px solid " + plot_color;
        std_label_selection.style.color = plot_color;
        } else {
        std_label_selection.style.backgroundColor = "#ffffff";
        std_label_selection.style.border = "2px solid #DADCE0";
        std_label_selection.style.color = "#000000";
        }
  } else {
      for (let i = 0; i < plot_selection.length-1; i++) {
      if (plot_selection[i].value !== plot_selected.value) {
          plot_selection[i].checked = false;
          label_selection[i].style.backgroundColor = "#ffffff";
          label_selection[i].style.border = "2px solid #DADCE0";
          label_selection[i].style.color = "#000000";
      }
      else {
          var plot_color = plot_colors[i+1]
          const [r, g, b] = plot_color.match(/\d+/g);
          const rgbaColor = `rgba(${r}, ${g}, ${b}, 0.2)`;
          plot_selected.checked = true;
          label_selection[i].style.backgroundColor = rgbaColor;
          label_selection[i].style.border = "2px solid " + plot_color;
          label_selection[i].style.color = plot_color;
          if (mean_plot_selection.checked) {
            mean_label_selection.style.backgroundColor = rgbaColor;
            mean_label_selection.style.border = "2px solid " + plot_color;
            mean_label_selection.style.color = plot_color;
            }
            else {
            mean_label_selection.style.backgroundColor = "#ffffff";
            mean_label_selection.style.border = "2px solid #DADCE0";
            mean_label_selection.style.color = "#000000";
            }
          if (std_plot_selection.checked) {
            std_label_selection.style.backgroundColor = rgbaColor;
            std_label_selection.style.border = "2px solid " + plot_color;
            std_label_selection.style.color = plot_color;
            } else {
            std_label_selection.style.backgroundColor = "#ffffff";
            std_label_selection.style.border = "2px solid #DADCE0";
            std_label_selection.style.color = "#000000";
            }
      }
      }
  }
  var slices_all = JSON.parse({{ get_slices(model_card)|safe|tojson }});
  var histories_all = JSON.parse({{ get_histories(model_card)|safe|tojson }});
  var thresholds_all = JSON.parse({{ get_thresholds(model_card)|safe|tojson }});
  var passed_all = JSON.parse({{ get_passed(model_card)|safe|tojson }});
  var names_all = JSON.parse({{ get_names(model_card)|safe|tojson }});
  var timestamps_all = JSON.parse({{ get_timestamps(model_card)|safe|tojson }});
  var sample_sizes_all = JSON.parse({{ get_sample_sizes(model_card)|safe|tojson }});

  for (let i = 0; i < selection.length; i++) {
      // use selection to set label_slice_selection background color
      for (let j = 0; j < inputs_all.length; j++) {
      if (inputs_all[j].name === selection[i].split(":")[0]) {
          if (inputs_all[j].value == selection[i].split(":")[1]) {
          inputs_all[j].checked = true;
          const [r, g, b] = plot_color.match(/\d+/g);
          const rgbaColor = `rgba(${r}, ${g}, ${b}, 0.2)`;
          label_slice_selection[j].style.backgroundColor = rgbaColor;
          label_slice_selection[j].style.border = "2px solid " + plot_color;
          label_slice_selection[j].style.color = plot_color;
          if (mean_plot_selection.checked) {
            mean_label_selection.style.backgroundColor = rgbaColor;
            mean_label_selection.style.border = "2px solid " + plot_color;
            mean_label_selection.style.color = plot_color;
            }
            else {
            mean_label_selection.style.backgroundColor = "#ffffff";
            mean_label_selection.style.border = "2px solid #DADCE0";
            mean_label_selection.style.color = "#000000";
            }
          if (std_plot_selection.checked) {
            std_label_selection.style.backgroundColor = rgbaColor;
            std_label_selection.style.border = "2px solid " + plot_color;
            std_label_selection.style.color = plot_color;
            } else {
            std_label_selection.style.backgroundColor = "#ffffff";
            std_label_selection.style.border = "2px solid #DADCE0";
            std_label_selection.style.color = "#000000";
            }
          }
          else {
          inputs_all[j].checked = false;
          label_slice_selection[j].style.backgroundColor = "#ffffff";
          label_slice_selection[j].style.border = "2px solid #DADCE0";
          label_slice_selection[j].style.color = "#000000";
          }
      }
      }
  }

  var radioGroups = {};
  var labelGroups = {};
  for (let i = 0; i < inputs_all.length; i++) {
      var input = inputs_all[i];
      var label = label_slice_selection[i];
      var groupName = input.name;
      if (!radioGroups[groupName]) {
      radioGroups[groupName] = [];
      labelGroups[groupName] = [];
      }
      radioGroups[groupName].push(input);
      labelGroups[groupName].push(label);
  }

  // use radioGroups to loop through selection changing only one element at a time
  for (let i = 0; i < selection.length; i++) {
      for (let j = 0; j < inputs_all.length; j++) {
      if (inputs_all[j].name === selection[i].split(":")[0]) {
          radio_group = radioGroups[selection[i].split(":")[0]];
          label_group = labelGroups[selection[i].split(":")[0]];
          for (let k = 0; k < radio_group.length; k++) {
          selection_copy = selection.slice();
          selection_copy[i] = selection[i].split(":")[0] + ":" + radio_group[k].value;
          // get idx of slices where all elements match
          var idx = Object.keys(slices_all).find(key => JSON.stringify(slices_all[key].sort()) === JSON.stringify(selection_copy.sort()));
          if (idx === undefined) {
              // set radio button to disabled and cursor to not allowed and color to gray if idx is undefined
              radio_group[k].disabled = true;
              label_group[k].style.cursor = "not-allowed";
              label_group[k].style.color = "gray";
              label_group[k].style.backgroundColor = "rgba(125, 125, 125, 0.2)";
          }
          else {
              radio_group[k].disabled = false;
              label_group[k].style.cursor = "pointer";
          }
          }
      }
      }
  }

  // Find the maximum sample size across all selections
  for (let i = 0; i < selections.length; i++) {
      if (selections[i] === null) {
      continue;
      }
      selection = selections[i]
      // get idx of slices where all elements match
      var idx = Object.keys(slices_all).find(key => JSON.stringify(slices_all[key].sort()) === JSON.stringify(selection));
      var sample_size_data = [];
      for (let i = 0; i < sample_sizes_all[idx].length; i++) {
      sample_size_data.push(sample_sizes_all[idx][i]);
      }
      maxSampleSize = Math.max(...sample_size_data);
  }

  traces = [];
  for (let i = 0; i < selections.length; i++) {
      if (selections[i] === null) {
      continue;
      }
      selection = selections[i]
      // get idx of slices where all elements match
      var idx = Object.keys(slices_all).find(key => JSON.stringify(slices_all[key].sort()) === JSON.stringify(selection));
      var history_data = [];
      for (let i = 0; i < histories_all[idx].length; i++) {
      history_data.push(parseFloat(histories_all[idx][i]));
      }
      var timestamp_data = [];
      for (let i = 0; i < timestamps_all[idx].length; i++) {
      // timestamp_data.push(timestamps_all[idx][i]);
      timestamp_data.push(formatDate(timestamps_all[idx][i]));
      }
      var sample_size_data = [];
      for (let i = 0; i < sample_sizes_all[idx].length; i++) {
      sample_size_data.push(sample_sizes_all[idx][i]);
      }
      var last_n_evals = document.getElementById("n_evals_slider_pot").value;
      history_data = history_data.slice(-last_n_evals);
      timestamp_data = timestamp_data.slice(-last_n_evals);
      sample_size_data = sample_size_data.slice(-last_n_evals);
      // get slope of line of best fit, if >0.01 then trending up, if <0.01 then trending down, else flat
      var slope = lineOfBestFit(history_data)[0];
      if (slope > 0.01) {
      var trend_keyword = "upwards";
      }
      else if (slope < -0.01) {
      var trend_keyword = "downwards";
      }
      else {
      var trend_keyword = "flat";
      }

      threshold = parseFloat(thresholds_all[idx]);
      passed = passed_all[idx];
      name = names_all[idx];

      // if passed is true set keyword to Above, if passed is false set keyword to Below
      if (passed) {
      var passed_keyword = "above";
      }
      else {
      var passed_keyword = "below";
      }

      // create title for plot: Current {metric name} is trending {trend_keyword} and is {passed_keyword} the threshold.
      // get number of nulls in selections, if 9 then plot title, else don't plot title
      var nulls = 0;
      for (let i = 0; i < selections.length; i++) {
      if (selections[i] === null) {
          nulls += 1;
      }
      }
      if (nulls === 10) {
      var plot_title = "Current " + name + " is trending " + trend_keyword + " and is " + passed_keyword + " the threshold.";
      var plot_title = multipleStringLines(plot_title);
      if (mean_plot_selection.checked || std_plot_selection.checked) {
        var showlegend = true;
      } else {
        var showlegend = false;
        }
      }
      else {
      var plot_title = "";
      var showlegend = true;
      }
      name = ""
      suffix = " ( "
      for (let i = 0; i < selection.length; i++) {
      if (selection[i].split(":")[0] === "metric") {
          name += selection[i].split(":")[1];
      }
      else {
          if (selection[i].split(":")[1].includes("overall")) {
          continue;
          } else {
          suffix += selection[i];
          suffix += ", ";
          }
      }
      }
      if (suffix === " ( ") {
      name += "";
      }
      else {
      suffix = suffix.slice(0, -2);
      name += suffix + " )";
      }
      if (nulls === 10) {
        var threshold_trace = {
        x: timestamp_data,
        y: Array.from({length: history_data.length}, (_, i) => threshold),
        mode: 'lines',
        type: 'scatter',
        marker: {color: 'rgb(0,0,0)'},
        line: {color: 'rgb(0,0,0)', dash: 'dot'},
        name: 'Threshold',
        showlegend: true,
        legendgroup: name + i,
        };
        traces.push(threshold_trace);
    }
        // Add sample size circles
        var sample_size_trace = {
          x: timestamp_data,
          y: history_data,
          mode: 'markers',
          marker: {
              sizemode: 'area',
              size: sample_size_data,
              sizeref: maxSampleSize / MAX_SIZE ** 2,
              color: `rgba(${plot_colors[i+1].slice(4, -1)}, 0.3)`,
              line: {width: 0},
          },
          text: sample_size_data.map((s, index) =>
              `Date: ${timestamp_data[index]}<br>Value: ${history_data[index].toFixed(2)}<br>Sample Size: ${s}`
          ),
          hoverinfo: 'text',
          hovertemplate: '%{text}<extra></extra>',
          name: name + ' (Sample Size)',
          legendgroup: name + i,
      };

      // Add main data points and line
      var main_trace = {
          x: timestamp_data,
          y: history_data,
          mode: 'lines+markers',
          type: 'scatter',
          marker: {
              color: plot_colors[i+1],
              symbol: 'circle',
          },
          line: {color: plot_colors[i+1]},
          name: name,
          legendgroup: name + i,
          hoverinfo: 'skip'
      };

      // check if length of history_data is >= mean_std_min_evals and if so get rolling mean and std if mean_plot_selection or std_plot_selection is checked
      var mean_std_min_evals = mean_plot_selection.value;
        if (history_data.length >= mean_std_min_evals ) {
            var history_mean_data = rollingMean(history_data, mean_std_min_evals);
            var history_std_data = rollingStd(history_data, mean_std_min_evals);
        }

        if (std_plot_selection.checked) {
            // shaded region for rolling std
            var trace_std_upper = {
                x: timestamp_data.slice(-history_std_data.length),
                y: history_mean_data.map((x, i) => x + history_std_data[i]),
                // fill: 'tonexty',
                fillcolor: `rgba(${plot_colors[i+1].slice(4, -1)}, 0.3)`,
                mode: 'lines',
                line: {width: 0, color: `rgba(${plot_colors[i+1].slice(4, -1)}, 0.3)`},
                type: 'scatter',
                showlegend: false,
                legendgroup: name + i,
                name: "Std. Dev. " + name,
            };
            var trace_std_lower = {
                x: timestamp_data.slice(-history_std_data.length),
                y: history_mean_data.map((x, i) => x - history_std_data[i]),
                fill: 'tonexty',
                fillcolor: `rgba(${plot_colors[i+1].slice(4, -1)}, 0.3)`,
                mode: 'none',
                type: 'scatter',
                name: "Std. Dev. " + name,
                legendgroup: name + i,
                };
            traces.push(trace_std_upper);
            traces.push(trace_std_lower);
        }
      if (mean_plot_selection.checked) {
        // dotted line for rolling mean
        var trace_mean = {
            x: timestamp_data.slice(-history_mean_data.length),
            y: history_mean_data,
            mode: 'lines',
            type: 'scatter',
            marker: {color: plot_colors[i+1]},
            line: {color: plot_colors[i+1], dash: 'dot'},
            name: "Mean " + name,
            legendgroup: name + i,
            };
          traces.push(trace_mean);
      }
      traces.push(sample_size_trace);
      traces.push(main_trace);

  }


  var width = Math.max(parent.innerWidth - 900, 500);
  var layout = {
      title: {
      text: plot_title,
      font: {
          family:  'Arial, Helvetica, sans-serif',
          size: 18,
      }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
      zeroline: false,
      showticklabels: true,
      showgrid: false,
      },
      yaxis: {
      gridcolor: '#ffffff',
      zeroline: false,
      showticklabels: true,
      showgrid: true,
      range: [-0.10, 1.10],
      },
      showlegend: showlegend,
      // show legend at top
      legend: {
      orientation: "h",
      yanchor: "bottom",
      y: -0.3,
      xanchor: "left",
      x: 0.1
      },
      margin: {
      l: 50,
      r: 50,
      b: 50,
      t: 50,
      pad: 4
      },
      // set height and width of plot to width of card minus 500px
      height: 500,
      width: width,
  }
  Plotly.newPlot(plot, traces, layout, {displayModeBar: false});
  }


function generate_model_card_plot() {
  var model_card_plots = []
  var overall_indices = {{overall_indices}}
  var histories = JSON.parse({{ get_histories(model_card)|safe|tojson }});
  var thresholds = JSON.parse({{ get_thresholds(model_card)|safe|tojson }});
  var timestamps = JSON.parse({{ get_timestamps(model_card)|safe|tojson }});

  for (let i = 0; i < overall_indices.length; i++) {
      var idx = overall_indices[i];
      var model_card_plot = "model-card-plot-" + idx;
      var threshold = thresholds[idx];
      var history_data = [];
      for (let i = 0; i < histories[idx].length; i++) {
      history_data.push(parseFloat(histories[idx][i]));
      }
      var timestamp_data = [];
      for (let i = 0; i < timestamps[idx].length; i++) {
      timestamp_data.push(timestamps[idx][i]);
      }
      var last_n_evals = document.getElementById("n_evals_slider_p").value;
      history_data = history_data.slice(-last_n_evals);
      timestamp_data = timestamp_data.slice(-last_n_evals);

      var model_card_fig = {
      data: [
          {
          x: timestamp_data,
          y: history_data,
          mode: "lines+markers",
          marker: { color: "rgb(31,111,235)" },
          line: { color: "rgb(31,111,235)" },
          showlegend: false,
          type: "scatter",
          name: ""
          },
          {
          x: timestamp_data,
          y: Array(history_data.length).fill(threshold),
          mode: "lines",
          line: { color: "black", dash: "dot" },
          showlegend: false,
          type: "scatter",
          name: ""
          }
      ],
      layout: {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          xaxis: {
          zeroline: false,
          showticklabels: true,
          showgrid: false,
          },
          yaxis: {
          gridcolor: "#ffffff",
          zeroline: false,
          showticklabels: true,
          showgrid: true,
          range: [-0.10, 1.10],
          },
          margin: { l: 30, r: 0, t: 0, b: 35 },
          padding: { l: 0, r: 0, t: 0, b: 0 },
          height: 150,
          width: 300
      }
      };
      if (history.length > 0) {
        Plotly.newPlot(model_card_plot, model_card_fig.data, model_card_fig.layout, {displayModeBar: false});
      }
    }
  }


function updatePlotSelection() {
  const inputs = document.querySelectorAll('#slice-selection input[type="radio"]:checked');
  var plot_selection = document.querySelectorAll('#plot-selection input[type="radio"]');
  var plot_selected = document.querySelectorAll('#plot-selection input[type="radio"]:checked')[0];
  // get number from value in plot_selected "Plot 1" -> 1
  var plot_number = parseInt(plot_selected.value.split(" ")[1]);
  var label_selection = document.querySelectorAll('#plot-selection label');
  var label_slice_selection = document.querySelectorAll('#slice-selection label');
  var button_plot_selection = document.querySelectorAll('#plot-selection button');

  var mean_std_plot_selection = document.querySelectorAll('#mean-std-selection input[type="checkbox"]');
  var mean_plot_selection = mean_std_plot_selection[0];
  var std_plot_selection = mean_std_plot_selection[1];

  var mean_std_label_selection = document.querySelectorAll('#mean-std-selection label');
  var mean_label_selection = mean_std_label_selection[0];
  var std_label_selection = mean_std_label_selection[1];

  // if plot_selected is "+" then add new radio button to plot_selection called "Plot N" where last plot is N-1 but keep "+" at end and set new radio button to checked for second last element
  if (plot_selected.value === "+") {
      // if 10 plots already exist, don't add new plot and gray out "+"
      if (plot_selection.length === 11) {
      plot_selected.checked = false;
      label_selection[label_selection.length-1].style.color = "gray";
      return;
      }
      // plot_name should be name of last plot + 1
      if (plot_selection.length === 2) {
      var plot_name = "Plot 2"
      } else {
      var plot_name = "Plot " + (parseInt(plot_selection[plot_selection.length - 2].value.split(" ")[1]) + 1);
      }
      var new_plot = document.createElement("input");
      new_plot.type = "radio";
      new_plot.id = plot_name;
      new_plot.name = "plot";
      new_plot.value = plot_name;
      new_plot.checked = true;
      var new_label = document.createElement("label");
      new_label.htmlFor = plot_name;
      new_label.innerHTML = plot_name;

      // Parse plot_color to get r, g, b values
      var plot_color = plot_colors[plot_selection.length]
      const [r, g, b] = plot_color.match(/\d+/g);
      const rgbaColor = `rgba(${r}, ${g}, ${b}, 0.2)`;
      // set background color of new radio button to plot_color
      new_label.style.backgroundColor = rgbaColor;
      new_label.style.border = "2px solid " + plot_color;
      new_label.style.color = plot_color;
      if (mean_plot_selection.checked) {
        mean_label_selection.style.backgroundColor = rgbaColor;
        mean_label_selection.style.border = "2px solid " + plot_color;
        mean_label_selection.style.color = plot_color;
        }
        else {
        mean_label_selection.style.backgroundColor = "#ffffff";
        mean_label_selection.style.border = "2px solid #DADCE0";
        mean_label_selection.style.color = "#000000";
        }
      if (std_plot_selection.checked) {
        std_label_selection.style.backgroundColor = rgbaColor;
        std_label_selection.style.border = "2px solid " + plot_color;
        std_label_selection.style.color = plot_color;
        } else {
        std_label_selection.style.backgroundColor = "#ffffff";
        std_label_selection.style.border = "2px solid #DADCE0";
        std_label_selection.style.color = "#000000";
        }

      // add button to delete plot
      var delete_button = document.createElement("button");
      delete_button.id = "button";
      delete_button.innerHTML = "&times";
      delete_button.style.backgroundColor = "transparent";
      delete_button.style.border = "none";
      new_label.style.padding = "1.5px 0px";
      new_label.style.paddingLeft = "10px";

      new_label.appendChild(delete_button)

      // make delete button from last plot invisible if not Plot 1
      if (plot_selection.length > 2) {
      button_plot_selection[button_plot_selection.length-1].style.visibility = "hidden";
      }
      // add on_click event to delete button and send plot number to deletePlotSelection
      delete_button.onclick = function() {deletePlotSelection(plot_number)};

      // insert new radio button and label before "+" radio button and after last radio button
      plot_selected.insertAdjacentElement("beforebegin", new_plot);
      plot_selected.insertAdjacentElement("beforebegin", new_label);

      // Add event listener to new radio button
      new_plot.addEventListener('change', updatePlotSelection);

      // set plot_selected to new plot
      var plot_selected = new_plot

      for (let i = 0; i < label_selection.length-1; i++) {
      plot_selection[i].checked = false;
      label_selection[i].style.backgroundColor = "#ffffff";
      label_selection[i].style.border = "2px solid #DADCE0";
      label_selection[i].style.color = "#000000";
      }

      selections[parseInt(plot_selected.value.split(" ")[1]-1)] = selections[parseInt(plot_selected.value.split(" ")[1]-2)]
      selection = selections[parseInt(plot_selected.value.split(" ")[1]-1)];
      plot_color = plot_colors[parseInt(plot_selected.value.split(" ")[1])];

      for (let i = 0; i < selection.length; i++) {
      // use selection to set label_slice_selection background color
      for (let j = 0; j < inputs_all.length; j++) {
          if (inputs_all[j].name === selection[i].split(":")[0]) {
          if (inputs_all[j].value == selection[i].split(":")[1]) {
              const [r, g, b] = plot_color.match(/\d+/g);
              const rgbaColor = `rgba(${r}, ${g}, ${b}, 0.2)`;
              inputs_all[j].checked = true;
              label_slice_selection[j].style.backgroundColor = rgbaColor;
              label_slice_selection[j].style.border = "2px solid " + plot_color;
              label_slice_selection[j].style.color = plot_color;
              if (mean_plot_selection.checked) {
                mean_label_selection.style.backgroundColor = rgbaColor;
                mean_label_selection.style.border = "2px solid " + plot_color;
                mean_label_selection.style.color = plot_color;
                }
                else {
                mean_label_selection.style.backgroundColor = "#ffffff";
                mean_label_selection.style.border = "2px solid #DADCE0";
                mean_label_selection.style.color = "#000000";
                }
              if (std_plot_selection.checked) {
                std_label_selection.style.backgroundColor = rgbaColor;
                std_label_selection.style.border = "2px solid " + plot_color;
                std_label_selection.style.color = plot_color;
                } else {
                std_label_selection.style.backgroundColor = "#ffffff";
                std_label_selection.style.border = "2px solid #DADCE0";
                std_label_selection.style.color = "#000000";
                }
          }
          else {
              inputs_all[j].checked = false;
              label_slice_selection[j].style.backgroundColor = "#ffffff";
              label_slice_selection[j].style.border = "2px solid #DADCE0";
              label_slice_selection[j].style.color = "#000000";
          }
          }
      }
      }
  } else {
      for (let i = 0; i < plot_selection.length-1; i++) {
      if (plot_selection[i].value !== plot_selected.value) {
          plot_selection[i].checked = false;
          label_selection[i].style.backgroundColor = "#ffffff";
          label_selection[i].style.border = "2px solid #DADCE0";
          label_selection[i].style.color = "#000000";
      }
      else {
          var plot_color = plot_colors[i+1]
          const [r, g, b] = plot_color.match(/\d+/g);
          const rgbaColor = `rgba(${r}, ${g}, ${b}, 0.2)`;
          plot_selected.checked = true;
          label_selection[i].style.backgroundColor = rgbaColor;
          label_selection[i].style.border = "2px solid " + plot_color;
          label_selection[i].style.color = plot_color;
          if (mean_plot_selection.checked) {
            mean_label_selection.style.backgroundColor = rgbaColor;
            mean_label_selection.style.border = "2px solid " + plot_color;
            mean_label_selection.style.color = plot_color;
            }
            else {
            mean_label_selection.style.backgroundColor = "#ffffff";
            mean_label_selection.style.border = "2px solid #DADCE0";
            mean_label_selection.style.color = "#000000";
            }
          if (std_plot_selection.checked) {
            std_label_selection.style.backgroundColor = rgbaColor;
            std_label_selection.style.border = "2px solid " + plot_color;
            std_label_selection.style.color = plot_color;
            } else {
            std_label_selection.style.backgroundColor = "#ffffff";
            std_label_selection.style.border = "2px solid #DADCE0";
            std_label_selection.style.color = "#000000";
            }
      }
      }
      selection = selections[parseInt(plot_selected.value.split(" ")[1]-1)];
      plot_color = plot_colors[parseInt(plot_selected.value.split(" ")[1])];
      for (let i = 0; i < selection.length; i++) {
      // use selection to set label_slice_selection background color
      for (let j = 0; j < inputs_all.length; j++) {
          if (inputs_all[j].name === selection[i].split(":")[0]) {
          if (inputs_all[j].value == selection[i].split(":")[1]) {
              inputs_all[j].checked = true;
              const [r, g, b] = plot_color.match(/\d+/g);
              const rgbaColor = `rgba(${r}, ${g}, ${b}, 0.2)`;
              label_slice_selection[j].style.backgroundColor = rgbaColor;
              label_slice_selection[j].style.border = "2px solid " + plot_color;
              label_slice_selection[j].style.color = plot_color;
              if (mean_plot_selection.checked) {
                mean_label_selection.style.backgroundColor = rgbaColor;
                mean_label_selection.style.border = "2px solid " + plot_color;
                mean_label_selection.style.color = plot_color;
                }
                else {
                mean_label_selection.style.backgroundColor = "#ffffff";
                mean_label_selection.style.border = "2px solid #DADCE0";
                mean_label_selection.style.color = "#000000";
                }
              if (std_plot_selection.checked) {
                std_label_selection.style.backgroundColor = rgbaColor;
                std_label_selection.style.border = "2px solid " + plot_color;
                std_label_selection.style.color = plot_color;
                } else {
                std_label_selection.style.backgroundColor = "#ffffff";
                std_label_selection.style.border = "2px solid #DADCE0";
                std_label_selection.style.color = "#000000";
                }
          }
          else {
              inputs_all[j].checked = false;
              label_slice_selection[j].style.backgroundColor = "#ffffff";
              label_slice_selection[j].style.border = "2px solid #DADCE0";
              label_slice_selection[j].style.color = "#000000";
          }
          }
      }
      }
  }
  var slices_all = JSON.parse({{ get_slices(model_card)|safe|tojson }});
  var histories_all = JSON.parse({{ get_histories(model_card)|safe|tojson }});
  var thresholds_all = JSON.parse({{ get_thresholds(model_card)|safe|tojson }});
  var passed_all = JSON.parse({{ get_passed(model_card)|safe|tojson }});
  var names_all = JSON.parse({{ get_names(model_card)|safe|tojson }});
  var timestamps_all = JSON.parse({{ get_timestamps(model_card)|safe|tojson }});
  var sample_sizes_all = JSON.parse({{ get_sample_sizes(model_card)|safe|tojson }});

  var radioGroups = {};
  var labelGroups = {};
  for (let i = 0; i < inputs_all.length; i++) {
      var input = inputs_all[i];
      var label = label_slice_selection[i];
      var groupName = input.name;
      if (!radioGroups[groupName]) {
      radioGroups[groupName] = [];
      labelGroups[groupName] = [];
      }
      radioGroups[groupName].push(input);
      labelGroups[groupName].push(label);
  }

  // use radioGroups to loop through selection changing only one element at a time
  for (let i = 0; i < selection.length; i++) {
      for (let j = 0; j < inputs_all.length; j++) {
      if (inputs_all[j].name === selection[i].split(":")[0]) {
          radio_group = radioGroups[selection[i].split(":")[0]];
          label_group = labelGroups[selection[i].split(":")[0]];
          for (let k = 0; k < radio_group.length; k++) {
          selection_copy = selection.slice();
          selection_copy[i] = selection[i].split(":")[0] + ":" + radio_group[k].value;
          // get idx of slices where all elements match
          var idx = Object.keys(slices_all).find(key => JSON.stringify(slices_all[key].sort()) === JSON.stringify(selection_copy.sort()));
          if (idx === undefined) {
              // set radio button to disabled and cursor to not allowed and color to gray if idx is undefined
              radio_group[k].disabled = true;
              label_group[k].style.cursor = "not-allowed";
              label_group[k].style.color = "gray";
              label_group[k].style.backgroundColor = "rgba(125, 125, 125, 0.2)";
          }
          else {
              radio_group[k].disabled = false;
              label_group[k].style.cursor = "pointer";
          }
          }
      }
      }
  }

  // Find the maximum sample size across all selections
  for (let i = 0; i < selections.length; i++) {
    if (selections[i] === null) {
    continue;
    }
    selection = selections[i]
    // get idx of slices where all elements match
    var idx = Object.keys(slices_all).find(key => JSON.stringify(slices_all[key].sort()) === JSON.stringify(selection));
    var sample_size_data = [];
    for (let i = 0; i < sample_sizes_all[idx].length; i++) {
    sample_size_data.push(sample_sizes_all[idx][i]);
    }
    maxSampleSize = Math.max(...sample_size_data);
}

  traces = [];
  var plot_number = parseInt(plot_selected.value.split(" ")[1]-1);
  for (let i = 0; i < selections.length; i++) {
      if (selections[i] === null) {
      continue;
      }
      selection = selections[i]

      // get idx of slices where all elements match
      var idx = Object.keys(slices_all).find(key => JSON.stringify(slices_all[key].sort()) === JSON.stringify(selection));
      var history_data = [];
      for (let i = 0; i < histories_all[idx].length; i++) {
      history_data.push(parseFloat(histories_all[idx][i]));
      }
      var timestamp_data = [];
      for (let i = 0; i < timestamps_all[idx].length; i++) {
      // timestamp_data.push(timestamps_all[idx][i]);
      timestamp_data.push(formatDate(timestamps_all[idx][i]));
      }
      var sample_size_data = [];
      for (let i = 0; i < sample_sizes_all[idx].length; i++) {
      sample_size_data.push(sample_sizes_all[idx][i]);
      }
      var last_n_evals = document.getElementById("n_evals_slider_pot").value;
      history_data = history_data.slice(-last_n_evals);
      timestamp_data = timestamp_data.slice(-last_n_evals);
      sample_size_data = sample_size_data.slice(-last_n_evals);

      // get slope of line of best fit, if >0.01 then trending up, if <0.01 then trending down, else flat
      var slope = lineOfBestFit(history_data)[0];
      if (slope > 0.01) {
      var trend_keyword = "upwards";
      }
      else if (slope < -0.01) {
      var trend_keyword = "downwards";
      }
      else {
      var trend_keyword = "flat";
      }

      threshold = parseFloat(thresholds_all[idx]);
      passed = passed_all[idx];
      name = names_all[idx];

      // if passed is true set keyword to Above, if passed is false set keyword to Below
      if (passed) {
      var passed_keyword = "above";
      }
      else {
      var passed_keyword = "below";
      }

      // create title for plot: Current {metric name} is trending {trend_keyword} and is {passed_keyword} the threshold.
      // get number of nulls in selections, if 9 then plot title, else don't plot title
      var nulls = 0;
      for (let i = 0; i < selections.length; i++) {
      if (selections[i] === null) {
          nulls += 1;
      }
      }
      if (nulls === 10) {
      var plot_title = "Current " + name + " is trending " + "flat" + " and is " + passed_keyword + " the threshold.";
      var plot_title = multipleStringLines(plot_title);
      if (mean_plot_selection.checked || std_plot_selection.checked) {
        var showlegend = true;
      } else {
        var showlegend = false;
        }
      }
      else {
      var plot_title = "";
      var showlegend = true;
      }
      name = ""
      suffix = " ( "
      for (let i = 0; i < selection.length; i++) {
      if (selection[i].split(":")[0] === "metric") {
          name += selection[i].split(":")[1];
      }
      else {
          if (selection[i].split(":")[1].includes("overall")) {
          continue;
          } else {
          suffix += selection[i];
          suffix += ", ";
          }
      }
      }
      if (suffix === " ( ") {
      name += "";
      }
      else {
      suffix = suffix.slice(0, -2);
      name += suffix + " )";
      }

      if (nulls === 10) {
        var threshold_trace = {
        x: timestamp_data,
        y: Array.from({length: history_data.length}, (_, i) => threshold),
        mode: 'lines',
        type: 'scatter',
        marker: {color: 'rgb(0,0,0)'},
        line: {color: 'rgb(0,0,0)', dash: 'dot'},
        name: 'Threshold',
        showlegend: true,
        legendgroup: name + i,
        };
        traces.push(threshold_trace);
    }

        // Add sample size circles
        var sample_size_trace = {
          x: timestamp_data,
          y: history_data,
          mode: 'markers',
          marker: {
              sizemode: 'area',
              size: sample_size_data,
              sizeref: maxSampleSize / MAX_SIZE ** 2,
              color: `rgba(${plot_colors[i+1].slice(4, -1)}, 0.3)`,
              line: {width: 0},
          },
          text: sample_size_data.map((s, index) =>
              `Date: ${timestamp_data[index]}<br>Value: ${history_data[index].toFixed(2)}<br>Sample Size: ${s}`
          ),
          hoverinfo: 'text',
          hovertemplate: '%{text}<extra></extra>',
          name: name + ' (Sample Size)',
          legendgroup: name + i,
      };

      // Add main data points and line
      var main_trace = {
          x: timestamp_data,
          y: history_data,
          mode: 'lines+markers',
          type: 'scatter',
          marker: {
              color: plot_colors[i+1],
              symbol: 'circle',
          },
          line: {color: plot_colors[i+1]},
          name: name,
          legendgroup: name + i,
          hoverinfo: 'skip'
      };

      // check if length of history_data is >= mean_std_min_evals and if so get rolling mean and std if mean_plot_selection or std_plot_selection is checked
      var mean_std_min_evals = mean_plot_selection.value;
        if (history_data.length >= mean_std_min_evals ) {
            var history_mean_data = rollingMean(history_data, mean_std_min_evals);
            var history_std_data = rollingStd(history_data, mean_std_min_evals);
        }
        if (std_plot_selection.checked) {
            // shaded region for rolling std
            var trace_std_upper = {
                x: timestamp_data.slice(-history_std_data.length),
                y: history_mean_data.map((x, i) => x + history_std_data[i]),
                fillcolor: `rgba(${plot_colors[i+1].slice(4, -1)}, 0.3)`,
                mode: 'lines',
                line: {width: 0, color: `rgba(${plot_colors[i+1].slice(4, -1)}, 0.3)`},
                type: 'scatter',
                showlegend: false,
                legendgroup: name + i,
                name: "Std. Dev. " + name,
            };
            var trace_std_lower = {
                x: timestamp_data.slice(-history_std_data.length),
                y: history_mean_data.map((x, i) => x - history_std_data[i]),
                fill: 'tonexty',
                fillcolor: `rgba(${plot_colors[i+1].slice(4, -1)}, 0.3)`,
                mode: 'none',
                type: 'scatter',
                name: "Std. Dev. " + name,
                legendgroup: name + i,
                };
            traces.push(trace_std_upper);
            traces.push(trace_std_lower);
        }
      if (mean_plot_selection.checked) {
        // dotted line for rolling mean
        var trace_mean = {
            x: timestamp_data.slice(-history_mean_data.length),
            y: history_mean_data,
            mode: 'lines',
            type: 'scatter',
            marker: {color: plot_colors[i+1]},
            line: {color: plot_colors[i+1], dash: 'dot'},
            name: "Mean " + name,
            legendgroup: name + i,
            };
          traces.push(trace_mean);
      }

    traces.push(main_trace);
    traces.push(sample_size_trace);
  }


  var width = Math.max(parent.innerWidth - 900, 500);
  var layout = {
      title: {
      text: plot_title,
      font: {
          family:  'Arial, Helvetica, sans-serif',
          size: 18,
      }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
      zeroline: false,
      showticklabels: true,
      showgrid: false,
      },
      yaxis: {
      gridcolor: '#ffffff',
      zeroline: false,
      showticklabels: true,
      showgrid: true,
      range: [-0.10, 1.10],
      },
      showlegend: showlegend,
      // show legend at top
      legend: {
      orientation: "h",
      yanchor: "bottom",
      y: -0.3,
      xanchor: "left",
      x: 0.1
      },
      margin: {
      l: 50,
      r: 50,
      b: 50,
      t: 50,
      pad: 4
      },
      // set height and width of plot to extra-wide to fit the plot
      height: 500,
      // get size of window and set width of plot to size of window
      width: width,
  }
  Plotly.newPlot(plot, traces, layout, {displayModeBar: false});
}

function deletePlotSelection(plot_number) {
  var plot_selection = document.querySelectorAll('#plot-selection input[type="radio"]');
  var label_selection = document.querySelectorAll('#plot-selection label');
  var label_slice_selection = document.querySelectorAll('#slice-selection label');
  var button_plot_selection = document.querySelectorAll('#plot-selection button');

  var mean_std_plot_selection = document.querySelectorAll('#mean-std-selection input[type="checkbox"]');
  var mean_plot_selection = mean_std_plot_selection[0];
  var std_plot_selection = mean_std_plot_selection[1];

  var mean_std_label_selection = document.querySelectorAll('#mean-std-selection label');
  var mean_label_selection = mean_std_label_selection[0];
  var std_label_selection = mean_std_label_selection[1];

  // set last plot to checked
  // get plot_selection with name "Plot N" where N is plot_number
  for (let i = 0; i < plot_selection.length; i++) {
    var plot_name = "Plot " + (plot_number+1)
    if (plot_selection[i].value === plot_name) {
      plot_number = i;
    }
  }
  plot_selection[plot_number].checked = false;
  plot_selection[plot_number-1].checked = true;

  // delete plot_selected and label
  plot_selection[plot_number].remove();
  label_selection[plot_number].remove();

  selections[plot_number] = null;

  // set selection to last plot
  selection = selections[plot_number-1];
  plot_color = plot_colors[plot_number-1];

  // set current plot selection color to plot_color
  const [r, g, b] = plot_color.match(/\d+/g);
  const rgbaColor = `rgba(${r}, ${g}, ${b}, 0.2)`;
  plot_selection[plot_number-1].style.backgroundColor = rgbaColor;
  plot_selection[plot_number-1].style.border = "2px solid " + plot_color;
  plot_selection[plot_number-1].style.color = plot_color;
  if (mean_plot_selection.checked) {
    mean_label_selection.style.backgroundColor = rgbaColor;
    mean_label_selection.style.border = "2px solid " + plot_color;
    mean_label_selection.style.color = plot_color;
    }
    else {
    mean_label_selection.style.backgroundColor = "#ffffff";
    mean_label_selection.style.border = "2px solid #DADCE0";
    mean_label_selection.style.color = "#000000";
    }
  if (std_plot_selection.checked) {
    std_label_selection.style.backgroundColor = rgbaColor;
    std_label_selection.style.border = "2px solid " + plot_color;
    std_label_selection.style.color = plot_color;
    } else {
    std_label_selection.style.backgroundColor = "#ffffff";
    std_label_selection.style.border = "2px solid #DADCE0";
    std_label_selection.style.color = "#000000";
    }

  // make visibility of delete button from last plot visible
  if (button_plot_selection.length >= 2) {
    button_plot_selection[button_plot_selection.length-2].style.visibility = "visible";
  }

  for (let i = 0; i < selection.length; i++) {
    // use selection to set label_slice_selection background color
    for (let j = 0; j < inputs_all.length; j++) {
      if (inputs_all[j].name === selection[i].split(":")[0]) {
        if (inputs_all[j].value == selection[i].split(":")[1]) {
          inputs_all[j].checked = true;
          label_slice_selection[j].style.backgroundColor = rgbaColor;
          label_slice_selection[j].style.border = "2px solid " + plot_color;
          label_slice_selection[j].style.color = plot_color;
        }
      }
    }
  }
  updatePlot();
}


// function to refresh plotly plots
function refreshPlotlyPlots() {
  const img_items = document.getElementsByClassName("img-item");
  for (let i = 0; i < img_items.length; i++) {
    if (img_items[i].getElementsByTagName("div").length > 0) {
      id = img_items[i].getElementsByClassName("plotly-graph-div")[0].id;
      var gd = document.getElementById(id);
      data = gd.data;
      layout = gd.layout;
      Plotly.update(id, data, layout);
    }
  }
}

function lineOfBestFit(y) {
  var x = Array.from({length: y.length}, (_, i) => i);
  var n = x.length;
  var x_sum = 0;
  var y_sum = 0;
  var xy_sum = 0;
  var xx_sum = 0;
  for (var i = 0; i < n; i++) {
      x_sum += x[i];
      y_sum += y[i];
      xy_sum += x[i] * y[i];
      xx_sum += x[i] * x[i];
  }
  var m = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum);
  var b = (y_sum - m * x_sum) / n;
  return [m, b];
  }

function rollingMean(data, window) {
    var mean = [];
    for (var i = 0; i < data.length - window + 1; i++) {
        var sum = 0;
        for (var j = 0; j < window; j++) {
        sum += data[i + j];
        }
        mean.push(sum / window);
    }
    return mean;
    }

function rollingStd(data, window) {
    var std = [];
    for (var i = 0; i < data.length - window + 1; i++) {
        var sum = 0;
        for (var j = 0; j < window; j++) {
        sum += data[i + j];
        }
        var mean = sum / window;
        var variance = 0;
        for (var j = 0; j < window; j++) {
        variance += (data[i + j] - mean) ** 2;
        }
        std.push(Math.sqrt(variance / window));
    }
    return std;
    }

  function formatDate(dateString) {
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');

    return `${year}-${month}-${day} ${hours}:${minutes}`;
  }
