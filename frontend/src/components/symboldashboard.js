/* eslint-disable react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { connect } from 'react-redux'
import { Segment, Header } from 'semantic-ui-react'
import SymbolChart from './symbolchart'
import SelectedTrade from './selectedtrade'

class SymbolDashboard extends React.Component {
    constructor(props) {
        super(props)
        this.handleChartClick = this.handleChartClick.bind(this)
        this.state = {
            selected: null,
            alert: null,
        }
    }

    handleChartClick(e) {
        const selected = e.currentItem
        if (selected != null) {
            this.setState({ selected })
            if (selected.flagged) {
                this.loadAlert(selected.id)
            }
        }
    }

    loadAlert(trade_pk) { // eslint-disable-line
        this.props.horizon('alerts')
        .find({ trade_pk })
        .fetch()
        .subscribe((alert) => {
            if (alert) {
                this.setState({ alert })
            }
        })
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
                <SelectedTrade trade={this.state.selected} alert={this.state.alert} />
            </div>
        )
    }
}

SymbolDashboard.propTypes = {
    symbol: PropTypes.string,
    trades: PropTypes.array,
    loadingError: PropTypes.bool,
    horizon: PropTypes.any,
}

SymbolDashboard.defaultProps = {
    symbol: '',
    trades: [],
    loadingError: false,
    horzon: null,
}

export default connect(
    state => ({
        horizon: state.db.horizon,
    })
)(SymbolDashboard)
