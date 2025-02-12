import logging

def setup_logger(name, level=logging.INFO):
    """Creates a basic logger that prints messages to the terminal."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console (stream) handler
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    # Add handler to logger
    logger.addHandler(stream_handler)

    return logger

# Example usage
if __name__ == "__main__":
    #logger = setup_logger('terminal_logger')
    #logger.info('This is an info message')
    pass
