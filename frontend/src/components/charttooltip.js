import React from 'react'
import { GenericChartComponent, tooltip, utils } from 'react-stockcharts'

const { ToolTipText, ToolTipTSpanLabel } = tooltip
const { functor } = utils

class ChartTooltip extends React.Component {
    constructor(props) {
        super(props)
        this.renderSVG = this.renderSVG.bind(this)
    }

    renderSVG(moreProps) {
        const { chartId, chartConfig, currentItem } = moreProps
        const { className, onClick, width, fontFamily, fontSize, origin: originProp, calculators, displayFormat } = this.props
        const { chartConfig: { height } } = moreProps
        const config = chartConfig


    }
}
