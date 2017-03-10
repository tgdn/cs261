/* eslint-disable react/sort-comp */

import React from 'react'
import { Menu, Modal, Header, Icon, Grid } from 'semantic-ui-react'

class HelpModal extends React.Component {
    state = {
        opened: false,
    }

    constructor(props) {
        super(props)
        this.handleOpen = this.handleOpen.bind(this)
        this.handleClose = this.handleClose.bind(this)
    }

    handleOpen() {
        this.setState({ opened: true, })
    }

    handleClose() {
        this.setState({ opened: false, })
    }

    render() {
        return (
            <Modal
                trigger={
                    <Menu.Item
                        onClick={this.handleOpen}
                    >
                        <Icon name='help' />
                    </Menu.Item>
                }
                open={this.state.opened}
                onClose={this.handleClose}
                size='small'
                closeIcon
            >
                <Header icon='help circle outline' content='Help' />
                <Modal.Content>
                    <Grid>
                        <Grid.Row columns={1}>
                            <Grid.Column>
                                <div>
                                    <Header size='small'>Home Page</Header>
                                    <p>
                                        Begin by uploading a CSV file or connect
                                        your system to a live feed through the
                                        settings menu in the top right hand corner.
                                    </p>
                                    <p>
                                        Once the trades have been imported,
                                        the left hand menu will display the different
                                        available stocks.
                                        <br />
                                        Select one to view the trades and anomalies
                                        for the day.
                                    </p>
                                </div>
                                <div>
                                    <br />
                                    <Header size='small'>Alert Page</Header>
                                    <p>
                                        The alerts page allows you to select
                                        the most important trades (ie they are classified by severity).
                                    </p>
                                    <p>
                                        Each alert will display a subset of
                                        the relevant trades for this anomaly.
                                    </p>
                                    <p>
                                        Each anomaly can be further classified by the user,
                                        its severity can be updated, and it
                                        can also be discarded if incorrect.
                                    </p>
                                </div>
                                <div>
                                    <br />
                                    <Header size='small'>Graphing Capabilities</Header>
                                    <p>
                                        The graphs can be zoomed in and out, panned
                                        laterally and vertically.
                                        <br />
                                        The x-axis and y-axis can be changed to
                                        display relevant data.
                                    </p>
                                    <p>
                                        Anomalies will show a red volume bar,
                                        clicking on one will allow you to review
                                        that anomaly in the alert page.
                                    </p>
                                </div>
                            </Grid.Column>
                        </Grid.Row>
                    </Grid>
                </Modal.Content>
            </Modal>
        )
    }
}

export default HelpModal
