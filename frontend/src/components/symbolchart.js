/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { utcParse, timeFormat, scaleTime } from 'd3-time-format'
import { format } from 'd3-format'
import { ChartCanvas, Chart, series, scale, coordinates, tooltip, axes, helper } from 'react-stockcharts'

const { BarSeries, LineSeries, AreaSeries, ScatterSeries, CircleMarker, SquareMarker, TriangleMarker } = series
const { discontinuousTimeScaleProvider } = scale

const { CrossHairCursor, MouseCoordinateX, MouseCoordinateY } = coordinates

const { HoverTooltip } = tooltip
const { XAxis, YAxis } = axes
const { fitWidth } = helper

const parseDatetime = utcParse('%Y-%m-%dT%H:%M:%S.%LZ')

class SymbolChart extends React.Component {
    constructor(props) {
        super(props)
        const { trades, minDatetime, maxDatetime } = this.parseData(this.props.trades)
        this.state = {
            trades,
            initialMinDatetime: minDatetime,
            initialMaxDatetime: maxDatetime
        }
    }

    shouldComponentUpdate(newProps, newState) {
        const latestTrade = this.state.trades[this.state.trades.length - 1].id
        const newLatestTrade = newState.trades[newState.trades.length - 1].id
        if (latestTrade !== newLatestTrade) {
            return true
        }
        return false
    }

    parseData(trades) {
        trades.forEach((d) => {
            d.datetime = new Date(parseDatetime(d.datetime)) // eslint-disable-line
        })
        const maxDatetime = trades[trades.length - 1].datetime
        let minDatetime = trades[0].datetime
        if (trades.length >= 40) {
            minDatetime = trades[trades.length - 40].datetime
        }
        return { trades, minDatetime, maxDatetime }
    }

    componentWillReceiveProps(newProps) {
        const { trades, minDatetime, maxDatetime } = this.parseData(newProps.trades)
        this.setState({
            trades,
            initialMinDatetime: minDatetime,
            initialMaxDatetime: maxDatetime
        })
    }

    render() {
        const { symbol, width, ratio } = this.props
        const { trades, initialMinDatetime, initialMaxDatetime } = this.state
        // <ScatterSeries
        //     yAccessor={d => d.price}
        //     marker={CircleMarker}
        //     markerProps={{ r: 3, stroke: '#878712', fill: '#C1C11F' }}
        // />
        return (
            <div>
                <ChartCanvas
                    ratio={ratio}
                    width={width}
                    height={600}
                    margin={{ left: 70, right: 70, top: 20, bottom: 50 }}
                    type='hybrid'
                    pointsPerPxThreshold={1}
                    seriesName={symbol}
                    data={trades}
                    xAccessor={d => d.datetime}
                    xScaleProvider={discontinuousTimeScaleProvider}
                    xExtents={[initialMinDatetime, initialMaxDatetime]}
                >
                    <Chart
                        id={1}
                        yExtents={d => d.price}
                        padding={{ top: 10, bottom: 20 }}
                        height={430}
                    >
                        <XAxis
                            axisAt='bottom'
                            orient='bottom'
                            opacity={0.5}
                            showTicks={false}
                            outerTickSize={0}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <YAxis
                            axisAt='right'
                            orient='right'
                            ticks={6}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <MouseCoordinateY
                            at="right"
                            orient="right"
                            displayFormat={format('.2f')}
                        />
                        <LineSeries
                            yAccessor={d => d.price}
                            stroke='#F4F42E'
                            strokeDasharray='Solid'
                        />
                    </Chart>
                    <Chart
                        id={2}
                        height={100}
                        yExtents={d => d.size}
                        origin={(w, h) => [0, h - 100]}
                    >
                        <XAxis
                            showGrid
                            axisAt='bottom'
                            orient='bottom'
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <YAxis
                            axisAt='left'
                            orient='left'
                            ticks={5}
                            tickFormat={format('.0s')}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <MouseCoordinateX
                            at="bottom"
                            orient="bottom"
                            displayFormat={timeFormat('%H:%M:%S.%L')}
                        />
                        <BarSeries
                            yAccessor={d => d.size}
                            fill='#6BA583'
                        />
                    </Chart>
                </ChartCanvas>
            </div>
        )
    }
}

export default fitWidth(SymbolChart)
