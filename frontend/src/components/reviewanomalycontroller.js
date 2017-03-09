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
    }

    static colorOrNa(shouldBe, current, color) {
        if (shouldBe === current) {
            return color
        }
        return 'grey'
    }

    handleSeverityChange(severity) {
        const { horizon: hz, alert } = this.props
        hz('alerts').update({
            id: alert.id,
            severity
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
            { key: 0, value: 'current', text: alert.description, selected: true },
            { key: 1, value: 'VS', text: 'Hourly volume spike' },
            { key: 2, value: 'VS-D', text: 'Volume spike over past day' },
            { key: 3, value: 'FFP', text: 'Fat finger error on price' },
            { key: 4, value: 'FFV', text: 'Fat finger error on volume' },
            { key: 6, value: 'PDBR', text: 'Hourly pump and dump/bear raid' },
            { key: 7, value: 'PDBR-D', text: 'Pump and dump/bear raid over past day' },
            { key: 8, value: 'other', text: 'Other' },
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
