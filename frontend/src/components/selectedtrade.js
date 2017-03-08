import React, { PropTypes } from 'react'
import { Segment, Statistic, Button } from 'semantic-ui-react'
import { format } from 'd3-format'
import { Link } from 'react-router'

const attrOrNa = (obj, attr, frmt) => {
    /* return 'n/a' if obj is undefined of if attr is unefined.
    -> format if required */
    if (obj == null || obj[attr] == undefined) { // eslint-disable-line
        return 'n/a'
    }
    return frmt ? format(frmt)(obj[attr]) : obj[attr]
}

const SelectedTrade = ({ trade: t }) => {
    let flagged = ''
    if (t != null) {
        flagged = t.flagged ? (
            <Button
                as={Link}
                to={`/flagged/${t.id}`}
                color='red'
                inverted
            >
                Review anomaly
            </Button>
        ) : ''
    }
    const bid = attrOrNa(t, 'bid', '.2f')
    const ask = attrOrNa(t, 'ask', '.2f')
    let bidAskColor = 'blue'
    let bidAskSpread = 'n/a'
    let color = 'blue'
    if (bid !== 'n/a' && ask !== 'n/a') {
        const rawBidAsk = (ask - bid) / ask
        color = (rawBidAsk < 0) ? 'red' : 'blue'
        bidAskSpread = format('.3%')((ask - bid) / ask)
    }
    const items = [
        { color: 'blue', label: 'Price (gbx)', value: attrOrNa(t, 'price', '.2f') },
        { color: 'blue', label: 'Vol', value: attrOrNa(t, 'size', '.4s') },
        { color: 'blue', label: 'Bid (gbx)', value: bid },
        { color: 'blue', label: 'Ask (gbx)', value: ask },
        { color, label: 'Bid-ask spread', value: bidAskSpread },
        { color: 'blue', value: flagged },
    ]
    return (
        <Segment attached='bottom' inverted>
            <Statistic.Group size='mini' widths={items.length}>
                {items.map((stat, i) => (
                    <Statistic inverted {...stat} key={i} />
                ))}
            </Statistic.Group>
        </Segment>
    )
}

SelectedTrade.propTypes = {
    trade: PropTypes.any,
}

SelectedTrade.defaultProps = {
    trade: null,
}

export default SelectedTrade
