/* eslint-disable react/forbid-prop-types, react/require-default-props */

import React, { PropTypes } from 'react'
import { format } from 'd3-format'
import { GenericChartComponent, tooltip, utils } from 'react-stockcharts'

const { ToolTipText } = tooltip
const { functor } = utils

/* tooltip text components */
const TooltipTSpanLabel = props => (
    <tspan className='purplechart-tooltip-label' text-anchor='right' fill='#999999' {...props}>{props.children}</tspan>
)

TooltipTSpanLabel.propTypes = {
    children: PropTypes.node.isRequired,
}

const TooltipTSpanValue = props => (
    <tspan className='purplechart-tooltip-value' fill='#333333' {...props}>{props.children}</tspan>
)

TooltipTSpanValue.propTypes = {
    children: PropTypes.node.isRequired,
}


class ChartTooltipElement extends React.Component {
    render() {
        const { fontFamily, fontSize, displayName, value } = this.props
        const [x, y] = this.props.origin
        const translate = `translate(${x}, ${y})`
        return (
            <g transform={translate}>
                <ToolTipText
                    x={5} y={0}
                    fontFamily={fontFamily}
                    fontSize={fontSize}
                >
                    <TooltipTSpanLabel>{displayName}</TooltipTSpanLabel>
                    <TooltipTSpanValue x='40' dy='0'>{value}</TooltipTSpanValue>
                </ToolTipText>
            </g>
        )
    }
}

ChartTooltipElement.propTypes = {
    origin: PropTypes.array.isRequired,
    displayName: PropTypes.string.isRequired,
    value: PropTypes.string.isRequired,
    fontFamily: PropTypes.string,
    fontSize: PropTypes.number,
    forChart: PropTypes.number.isRequired,
    options: PropTypes.object.isRequired,
}

ChartTooltipElement.defaultProps = {
    options: {}
}

class ChartTooltip extends React.Component {
    constructor(props) {
        super(props)
        this.renderSVG = this.renderSVG.bind(this)
    }

    renderSVG(moreProps) {
        const { chartId, chartConfig, currentItem } = moreProps

        const { className, width, fontFamily, fontSize, origin: originProp } = this.props
        const { chartConfig: { height } } = moreProps

        const config = chartConfig

        const origin = functor(originProp)
        const [x, y] = origin(width, height)
        const [ox, oy] = config.origin

        const price = currentItem ? currentItem.price.toString() : 'n/a'
        const volume = currentItem ? currentItem.size.toString() : 'n/a'

        return (
            <g
                transform={`translate(${ ox + x }, ${ oy + y })`}
                className={className}
                fill='#FFFFFF'
            >
                <ChartTooltipElement
                    origin={[0, 0]}
                    displayName='Price'
                    value={price}
                    forChart={chartId}
                    fontFamily={fontFamily}
                    fontSize={fontSize}
                />
                <ChartTooltipElement
                    origin={[0, 15]}
                    displayName='Vol'
                    value={volume}
                    forChart={chartId}
                    fontFamily={fontFamily}
                    fontSize={fontSize}
                />
            </g>
        )
    }

    render = () => (
        <GenericChartComponent
            clip={false}
            svgDraw={this.renderSVG}
            drawOnMouseMove
        />
    )
}

ChartTooltip.propTypes = {
    className: PropTypes.string,
    origin: PropTypes.array.isRequired,
    fontFamily: PropTypes.string,
    fontSize: PropTypes.number,
    width: PropTypes.number,
    forDataSeries: PropTypes.arrayOf(PropTypes.number),
}

ChartTooltip.defaultProps = {
    className: 'react-stockcharts-toottip react-stockcharts-moving-average-tooltip',
    displayFormat: format('.2f'),
    origin: [0, 10],
    width: 65,
}

export default ChartTooltip
