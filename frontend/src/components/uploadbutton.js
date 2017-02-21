import React, { PropTypes } from 'react'

class UploadButton extends React.Component {
    /* eslint-disable react/sort-comp */
    style = {
        width: '0.1px',
        height: '0.1px',
        opacity: '0',
        overflow: 'hidden',
        position: 'absolute',
        zIndex: '-1',
    }

    state = {
        uploading: false,
    }
    /* eslint-enable */

    constructor(props) {
        super(props)
        this.handleChange = this.handleChange.bind(this)
    }

    handleChange(e) {
        const input = e.target
        const files = input.files
        const data = new FormData() // eslint-disable-line no-undef
        const { acceptedTypes } = this.props

        if (files.length === 0) {
            return
        }
        this.setState({ uploading: true, })

        for (let i = 0; i < files.length; i++) { // eslint-disable-line no-plusplus
            const file = files[i]

            if (file.type && acceptedTypes.indexOf(file.type) !== -1) {
                data.append('file[]', file)
            } else {
                this.props.handleFailure(this.props.typeErrorMessage)
                break
            }
        }
        this.handleUpload(data)
    }

    handleUpload(data) {
        const that = this
        const { extraData } = this.props
        for (let i = 0; i < extraData.length; i++) { // eslint-disable-line no-plusplus
            data.append(extraData[i].key, extraData[i].value)
        }

        const xhr = new XMLHttpRequest() // eslint-disable-line no-undef
        xhr.onprogress = (event) => {
            if (event.lengthComputable) {
                const sent = event.loaded / event.total
                that.props.handleProgress(sent)
            }
        }

        xhr.onreadystatechange = () => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let res = xhr.responseText
                /* eslint-disable no-empty */
                try {
                    res = JSON.parse(res)
                } catch (e) {
                } finally {
                    that.props.handleSuccess(res)
                }
                // eslint-enable */
            } else if (xhr.readyState === 4 && xhr.status !== 200) {
                that.props.handleFailure('An unexpected error occurred, try again')
            }

            // update state whenever its finished
            if (xhr.readyState === 4) {
                that.setState({ uploading: false, })
            }
        }

        xhr.onerror = () => {
            that.props.handleFailure('Error while sending file')
        }

        xhr.open('POST', this.props.uploadTo)
        xhr.setRequestHeader('Cache-Control', 'no-cache')
        xhr.send(data)
    }

    render() {
        const {
            id,
            acceptedTypes,
            className,
            multiple,
            uploadingContent,
            children,
        } = this.props

        return (
            <label
                class={className + (this.state.uploading ? ' disabled' : '')}
                htmlFor={id}
            >
                <input
                    id={id}
                    type='file'
                    disabled={this.state.uploading}
                    accept={acceptedTypes.join(',')}
                    style={this.style}
                    multiple={multiple}
                    onChange={this.handleChange}
                />
                <span>
                    {this.state.uploading ? uploadingContent : children}
                </span>
            </label>
        )
    }
}

UploadButton.propTypes = {
    id: PropTypes.string,
    children: PropTypes.node,
    className: PropTypes.string,
    multiple: PropTypes.bool,
    uploadingContent: PropTypes.node,
    acceptedTypes: PropTypes.array.isRequired, // eslint-disable-line
    uploadTo: PropTypes.string.isRequired,
    extraData: PropTypes.arrayOf(
        PropTypes.shape({
            key: PropTypes.string,
            value: PropTypes.string,
        })
    ),
    typeErrorMessage: PropTypes.string,
    handleFailure: PropTypes.func,
    handleSuccess: PropTypes.func,
    handleProgress: PropTypes.func,
}

UploadButton.defaultProps = {
    id: '',
    children: null,
    className: '',
    multiple: false,
    acceptedTypes: [],
    extraData: [],
    typeErrorMessage: 'The file type is not valid',
    uploadingContent: 'Uploading...',
    handleFailure: () => {},
    handleSuccess: () => {},
    handleProgress: () => {}
}

export default UploadButton
