/* eslint-disable react/forbid-prop-types, react/sort-comp, react/require-default-props */

import React, { PropTypes } from 'react'
import { utcParse, timeFormat } from 'd3-time-format'
import { format, formatDefaultLocale } from 'd3-format'
import {
    ChartCanvas,
    Chart,
    series,
    scale,
    coordinates,
    interactive,
    axes,
    helper
} from 'react-stockcharts'

import ChartTooltip from './charttooltip'

const { BarSeries, LineSeries } = series
const { discontinuousTimeScaleProvider } = scale

const {
    CrossHairCursor,
    MouseCoordinateX,
    MouseCoordinateY,
    CurrentCoordinate,
    EdgeIndicator
} = coordinates

const { ClickCallback } = interactive
const { XAxis, YAxis } = axes
const { fitWidth } = helper

const parseDatetime = utcParse('%Y-%m-%dT%H:%M:%S.%LZ')

formatDefaultLocale({
    decimal: '.',
    thousands: ',',
    grouping: [3],
    currency: ['£', '']
})

class SymbolChart extends React.Component {
    constructor(props) {
        super(props)
        const { trades, minDatetime, maxDatetime, maxVolume } = this.parseData(this.props.trades)
        this.state = {
            trades,
            maxVolume,
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

    parseData(trades) { // eslint-disable-line
        let maxVolume = 0
        trades.forEach((d) => {
            d.datetime = new Date(parseDatetime(d.datetime)) // eslint-disable-line
            if (d.size > maxVolume) {
                maxVolume = d.size
            }
        })
        /* try and get best minimum datetime */
        const maxDatetime = trades[trades.length - 1].datetime
        let minDatetime = trades[0].datetime
        if (trades.length >= 100) {
            minDatetime = trades[trades.length - 100].datetime
        } else if (trades.length >= 70) {
            minDatetime = trades[trades.length - 70].datetime
        } else if (trades.length >= 40) {
            minDatetime = trades[trades.length - 40].datetime
        }
        return { trades, minDatetime, maxDatetime, maxVolume }
    }

    componentWillReceiveProps(newProps) {
        const { trades, minDatetime, maxDatetime, maxVolume } = this.parseData(newProps.trades)
        this.setState({
            trades,
            maxVolume,
            initialMinDatetime: minDatetime,
            initialMaxDatetime: maxDatetime,
        })
    }

    render() {
        const { symbol, width, ratio, flagAnomalies, flagOne } = this.props
        const { trades, initialMinDatetime, initialMaxDatetime } = this.state

        /* chart settings */
        const margin = { left: 70, right: 80, top: 20, bottom: 50 }
        const height = 430
        const gridHeight = height - margin.top - margin.bottom
        const gridWidth = width - margin.left - margin.right

        /* define grid */
        const yGrid = {
            innerTickSize: -1 * gridWidth,
            tickStrokeDasharray: 'Solid',
            tickStrokeOpacity: 0.1,
            tickStrokeWidth: 0.2
        }
        const xGrid = {
            innerTickSize: -1 * gridHeight,
            tickStrokeDasharray: 'Solid',
            tickStrokeOpacity: 0.1,
            tickStrokeWidth: 0.2
        }

        return (
            <div>
                <ChartCanvas
                    ratio={ratio}
                    width={width}
                    height={height}
                    margin={margin}
                    type='svg'
                    pointsPerPxThreshold={4}
                    seriesName={symbol}
                    data={trades}
                    xAccessor={d => d.datetime}
                    xScaleProvider={discontinuousTimeScaleProvider}
                    xExtents={[initialMinDatetime, initialMaxDatetime]}
                >
                    <Chart
                        yPan
                        id={1}
                        yExtents={d => d.price}
                        padding={{ top: 10, bottom: 0 }}
                        height={290}
                    >
                        <XAxis
                            axisAt='bottom'
                            orient='bottom'
                            opacity={0.4}
                            showTicks={false}
                            outerTickSize={0}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                            {...xGrid}
                        />
                        <YAxis
                            axisAt='right'
                            orient='right'
                            ticks={6}
                            fontSize={11}
                            tickFormat={format('.2f')}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                            {...yGrid}
                        />
                        <MouseCoordinateY
                            at="right"
                            orient="right"
                            displayFormat={format('.2f')}
                        />
                        <LineSeries
                            yAccessor={d => d.price}
                            stroke='#d942f4'
                            opacity={0.3}
                            strokeDasharray='Solid'
                        />
                        <CurrentCoordinate
                            yAccessor={d => d.price}
                            stroke='#4e91fc'
                        />
                        <EdgeIndicator
                            itemType="last"
                            orient="right"
                            edgeAt="right"
                            yAccessor={d => d.size}
                            displayFormat={format('$.2s')}
                            fill="#4e91fc"
                        />
                        <ClickCallback enabled onClick={this.props.handleClick} />
                        <ChartTooltip />
                    </Chart>
                    <Chart
                        id={2}
                        height={100}
                        yExtents={() => [0, this.state.maxVolume * 10]}
                        origin={(w, h) => [0, h - 70]}
                    >
                        <XAxis
                            showGrid
                            axisAt='bottom'
                            orient='bottom'
                            fontSize={11}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                            flexTicks
                        />
                        <YAxis
                            axisAt='left'
                            orient='left'
                            ticks={5}
                            tickFormat={format('.0s')}
                            fontSize={11}
                            tickStroke='#FFFFFF'
                            stroke='#FFFFFF'
                        />
                        <MouseCoordinateX
                            at="bottom"
                            orient="bottom"
                            displayFormat={timeFormat('%H:%M:%S')}
                        />
                        <MouseCoordinateY
                            at="right"
                            orient="right"
                            displayFormat={format('.4s')}
                        />
                        <BarSeries
                            yAccessor={d => d.size}
                            fill={d => {
                                if (flagAnomalies) {
                                    return d.flagged ? '#e54444' : '#f9fc4e'
                                } else if (flagOne !== null) {
                                    return (d.id === flagOne) ? '#d942f4' : '#f9fc4e'
                                }
                                return '#f9fc4e'
                            }}
                        />
                    </Chart>
                    <CrossHairCursor stroke='#BBBBBB' />
                </ChartCanvas>
            </div>
        )
    }
}

SymbolChart.propTypes = {
    symbol: PropTypes.string,
    width: PropTypes.number,
    ratio: PropTypes.number,
    handleClick: PropTypes.func.isRequired,
    trades: PropTypes.array,
    flagAnomalies: PropTypes.bool,
    flagOne: PropTypes.number,
}

SymbolChart.defaultProps = {
    handleClick: () => {},
    trades: [],
    flagAnomalies: true,
}


export default fitWidth(SymbolChart)
