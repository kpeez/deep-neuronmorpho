# Viewing TensorBoard on a remote server

To view tensorboard on a remote server, you need to set up a tunnel to the server. This can be achieved by the following:

1. On the remote machine, run:

    ```bash
    tensorboard --logdir <path> --port 6006
    ```

2. On the local machine, run:

    ```bash
    ssh -N -f -L localhost:16006:localhost:6006 <user@remote>
    ```

3. Lastly, navigate to `http://localhost:16006` to view tensorboard.
