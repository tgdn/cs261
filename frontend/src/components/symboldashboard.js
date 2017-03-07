/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { Segment, Header } from 'semantic-ui-react'
import SymbolChart from './symbolchart'
import SelectedTrade from './selectedtrade'

class SymbolDashboard extends React.Component {
    constructor(props) {
        super(props)
        this.handleChartClick = this.handleChartClick.bind(this)
        this.state = {
            selected: null
        }
    }

    handleChartClick(e) {
        if (e.currentItem != null) {
            this.setState({ selected: e.currentItem })
        }
    }

    render() {
        const { symbol, trades } = this.props
        return (
            <div>
                <Segment inverted attached>
                    {trades.length === 0 ? (
                        <Header inverted as='h2'>No data to work with</Header>
                    ) : (
                        <SymbolChart
                            symbol={symbol}
                            trades={trades}
                            handleClick={this.handleChartClick}
                        />
                    )}
                </Segment>
                <SelectedTrade trade={this.state.selected} />
            </div>
        )
    }
}

SymbolDashboard.propTypes = {
    symbol: PropTypes.string,
    trades: PropTypes.array,
    loadingError: PropTypes.bool,
}

SymbolDashboard.defaultProps = {
    symbol: '',
    trades: [],
    loadingError: false,
}

export default SymbolDashboard
