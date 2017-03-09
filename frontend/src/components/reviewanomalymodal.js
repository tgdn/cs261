/* eslint-disable react/sort-comp, react/forbid-prop-types */

import React, { PropTypes } from 'react'
import { Button, Modal, Header, Icon } from 'semantic-ui-react'
import ReviewAnomalyController from './reviewanomalycontroller'

class ReviewAnomalyModal extends React.Component {
    constructor(props) {
        super(props)
        this.handleOpen = this.handleOpen.bind(this)
        this.handleClose = this.handleClose.bind(this)
        this.state = {
            opened: false,
        }
    }

    handleOpen = () => this.setState({ opened: true })
    handleClose = () => this.setState({ opened: false })

    render = () => (
        <Modal
            trigger={
                <Button
                    inverted
                    onClick={this.handleOpen}
                >
                    Review
                </Button>
            }
            open={this.state.opened}
            onClose={this.handleClose}
            size='small'
            closeIcon
        >
            <Header icon='browser' content='Review anomaly' />
            <Modal.Content>
                <ReviewAnomalyController alert={this.props.alert} />
            </Modal.Content>
        </Modal>
    )
}

ReviewAnomalyModal.propTypes = {
    alert: PropTypes.object.isRequired,
}

export default ReviewAnomalyModal
