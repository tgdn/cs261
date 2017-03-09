/* eslint-disable react/sort-comp, react/prop-types, react/forbid-prop-types */

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
    // constructor(props) {
    //     super(props)
    // }

    render() {
        const alert = this.props.alert
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
                                <Button color='red'>
                                    1
                                </Button>
                                <Button color='yellow'>
                                    2
                                </Button>
                                <Button color='blue'>
                                    3
                                </Button>
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
