/* eslint-disable
   react/sort-comp,
   react/prop-types,
   react/forbid-prop-types,
   react/no-array-index-key
  */

import React, { PropTypes } from 'react'
import { connect } from 'react-redux'
import {
    Container,
    Grid,
    Button,
    Header,
    Dropdown,
    Divider,
    Confirm,
    Dimmer,
    Loader
} from 'semantic-ui-react'

class ReviewAnomalyController extends React.Component {
    constructor(props) {
        super(props)
        this.handleSeverityChange = this.handleSeverityChange.bind(this)
        this.handleDescriptionChange = this.handleDescriptionChange.bind(this)
    }

    static colorOrNa(shouldBe, current, color) {
        if (shouldBe === current) {
            return color
        }
        return 'grey'
    }

    descriptionMaker(type) {
        const { symbol } = this.props.alert
        switch (type) {
        case 'VS':
            return `Hourly volume spike for ${symbol}`
        case 'VS-D':
            return `Volume spike over past day for ${symbol}`
        case 'PDBR':
            return `Hourly pump and dump/bear raid for ${symbol}`
        case 'PDBR-D':
            return `Pump and dump/bear raid over past day for ${symbol}`
        case 'FFP':
            return `Fat finger error on price for ${symbol}`
        case 'FFV':
            return `Fat finger error on volume for ${symbol}`
        case 'other':
        default:
            return `Other type of anomaly for ${symbol}`
        }
    }

    handleSeverityChange(severity) {
        const { horizon: hz, alert } = this.props
        hz('alerts').update({
            id: alert.id,
            severity
        })
    }

    handleDescriptionChange(e, data) {
        const { value } = data
        const { horizon: hz, alert } = this.props
        if (value === 'current') { return } // early return
        hz('alerts').update({
            id: alert.id,
            description: this.descriptionMaker(value)
        })
    }

    render() {
        const alert = this.props.alert
        const severityOptions = [
            { val: 1, color: ReviewAnomalyController.colorOrNa(1, alert.severity, 'red') },
            { val: 2, color: ReviewAnomalyController.colorOrNa(2, alert.severity, 'yellow') },
            { val: 3, color: ReviewAnomalyController.colorOrNa(3, alert.severity, 'blue') },
        ]
        const ANOMALY_OPTIONS = [
            { key: 0, value: 'current', text: `${alert.description} (current)`, selected: true },
            { key: 1, value: 'VS', text: this.descriptionMaker('VS') },
            { key: 2, value: 'VS-D', text: this.descriptionMaker('VS-D') },
            { key: 3, value: 'FFP', text: this.descriptionMaker('FFP') },
            { key: 4, value: 'FFV', text: this.descriptionMaker('FFV') },
            { key: 6, value: 'PDBR', text: this.descriptionMaker('PDBR') },
            { key: 7, value: 'PDBR-D', text: this.descriptionMaker('PDBR-D') },
            { key: 8, value: 'other', text: this.descriptionMaker('other') },
        ]
        return (
            <Grid>
                <Grid.Row columns={2}>
                    <Grid.Column>
                        <div>
                            <Header size='small'>Severity</Header>
                            <Button.Group>
                                {severityOptions.map((el, i) => (
                                    <Button
                                        key={i}
                                        color={el.color}
                                        onClick={() => { this.handleSeverityChange(el.val) }}
                                    >
                                        {el.val}
                                    </Button>
                                ))}
                            </Button.Group>
                        </div>
                    </Grid.Column>
                    <Grid.Column>
                        <div>
                            <Header size='small'>Type</Header>
                            <Dropdown
                                placeholder={alert.description}
                                fluid
                                search
                                selection
                                onChange={this.handleDescriptionChange}
                                options={ANOMALY_OPTIONS}
                            />
                        </div>
                    </Grid.Column>
                </Grid.Row>
            </Grid>
        )
    }
}

ReviewAnomalyController.propTypes = {
    alert: PropTypes.object.isRequired,
}

export default connect(
    state => ({
        horizon: state.db.horizon
    })
)(ReviewAnomalyController)
