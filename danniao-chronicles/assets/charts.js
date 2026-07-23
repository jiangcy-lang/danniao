// charts.js — 丹鸟数字生命编年史 图表
(function() {
  // 初始化 Mermaid
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#16171d',
        primaryColor: '#1c1d25',
        primaryTextColor: '#e8e3d8',
        primaryBorderColor: '#d4a04a',
        lineColor: '#4ec0b4',
        secondaryColor: '#1c1d25',
        tertiaryColor: '#16171d',
        fontSize: '14px'
      },
      securityLevel: 'loose'
    });
  }

  // 读取CSS变量
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // --- 图表: 测试数量随版本增长趋势 ---
  var chartEl = document.getElementById('chart-test-growth');
  if (chartEl && typeof echarts !== 'undefined') {
    var chart = echarts.init(chartEl, null, { renderer: 'svg' });

    var versions = ['Step 4A\n(07-22下午)', 'v0.2.0\n(07-23上午)', 'v0.3.0\n(07-23中午)', 'v0.4.0\n(07-23下午)', 'v0.4.1\n(07-23傍晚)'];
    var testCounts = [16, 55, 121, 133, 133];
    var newTests = [16, 39, 66, 12, 0];

    chart.setOption({
      animation: false,
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        appendToBody: true,
        backgroundColor: bg2,
        borderColor: rule,
        textStyle: { color: ink, fontSize: 13 },
        formatter: function(params) {
          var html = '<div style="font-weight:700;margin-bottom:6px;color:' + accent + '">' + params[0].name.replace('\n', ' ') + '</div>';
          params.forEach(function(p) {
            html += '<div style="margin:2px 0"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:' + p.color + ';margin-right:6px"></span>' + p.seriesName + ': <strong>' + p.value + '</strong></div>';
          });
          return html;
        }
      },
      legend: {
        data: ['累计测试', '新增测试'],
        textStyle: { color: muted, fontSize: 12 },
        top: 5,
        itemGap: 20
      },
      grid: {
        left: '8%',
        right: '5%',
        top: '18%',
        bottom: '15%'
      },
      xAxis: {
        type: 'category',
        data: versions,
        axisLine: { lineStyle: { color: rule } },
        axisLabel: {
          color: muted,
          fontSize: 11,
          interval: 0,
          lineHeight: 14
        },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisLabel: { color: muted, fontSize: 11 },
        splitLine: { lineStyle: { color: rule, type: 'dashed', opacity: 0.5 } },
        max: 150
      },
      series: [
        {
          name: '累计测试',
          type: 'bar',
          data: testCounts,
          itemStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: accent },
                { offset: 1, color: 'rgba(212,160,74,0.3)' }
              ]
            },
            borderRadius: [6, 6, 0, 0]
          },
          barWidth: '35%',
          label: {
            show: true,
            position: 'top',
            color: accent,
            fontSize: 14,
            fontWeight: 700,
            formatter: '{c}'
          },
          z: 2
        },
        {
          name: '新增测试',
          type: 'line',
          data: newTests,
          smooth: true,
          symbol: 'circle',
          symbolSize: 8,
          itemStyle: { color: accent2 },
          lineStyle: { width: 2, color: accent2 },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(78,192,180,0.25)' },
                { offset: 1, color: 'rgba(78,192,180,0)' }
              ]
            }
          },
          z: 1
        }
      ]
    });

    window.addEventListener('resize', function() { chart.resize(); });
  }
})();
