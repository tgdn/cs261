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
    const items = [
        { label: 'Price (gbx)', value: attrOrNa(t, 'price', '.2f') },
        { label: 'Vol', value: attrOrNa(t, 'size', '.4s') },
        { label: 'Bid (gbx)', value: attrOrNa(t, 'bid', '.2f') },
        { label: 'Ask (gbx)', value: attrOrNa(t, 'ask', '.2f') },
        { value: flagged },
    ]
    return (
        <Segment attached='bottom' inverted>
            <Statistic.Group
                inverted
                size='mini'
                items={items}
                color='blue'
                widths={5}
            />
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
