import matplotlib.animation as animation
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import random as rand

from sympy import fps


# Generate a signal that is a sum of two semi-random signals
def generate_signal():
    # Generate two random signals
    signal1 = np.array([rand.random() for i in range(100)])
    signal2 = np.array([rand.random() for i in range(100)])

    # Generate random shifts
    freq_shift = rand.randint(1, 30)
    amp_shift = rand.randint(1, 2)
    phase_shift = rand.randint(1, 6)

    # Add the two signals together
    signal = signal1 + signal2

    # Shift the signal
    signal = np.roll(signal, freq_shift)

    # amplify the signal
    signal = signal * amp_shift

    # phase shift the signal
    signal = signal * np.exp(1j * phase_shift)

    # return details of the signal (specifically the frequency shifts, amplitude, and phase shifts)
    return signal, freq_shift, amp_shift, phase_shift


# Function to calculate the PSD of a signal using Welch's method
def welch_psd(a_signal, sig_length):
    # Calculate the PSD using Welch's method
    f, psd = signal.welch(a_signal, fs=1, nperseg=sig_length)

    # Return the PSD
    return psd

# Function to calculate the PSD of a signal using the FFT


def fft_psd(a_signal, sig_length):
    # Calculate the PSD using the FFT
    f, psd = signal.periodogram(a_signal, fs=1, nfft=sig_length)
    # Return the PSD
    return psd


# Function to plot a plain set of two signals
def plot_signal(signal):
    # Plot the signal
    plt.plot(signal)

    # Set the labels
    plt.ylabel('Signal')
    plt.xlabel('Time')

    # Add a grid to the plot
    plt.grid()

    # Tighten the layout
    plt.tight_layout()

    # Save the plots
    plt.savefig('raw_signal.png')

    # Display the plots
    plt.show()


# Function for plotting the PSDs without markers
def plot_without_markers(welch_psd_vals, fft_psd_vals):

    # Plot the two PSDs on the same plot
    plt.plot(welch_psd_vals, 'b', label='Welch')
    plt.plot(fft_psd_vals, 'r', label='FFT')

    # Set the labels
    plt.ylabel('PSD')
    plt.xlabel('Frequency')

    # Add a legend and set the location to the upper right
    plt.legend(loc='upper right')

    # Add a title to the plot
    plt.title('PSD Comparison')

    # Add a grid to the plot
    plt.grid()

    # Tighten the layout
    plt.tight_layout()

    # Save the plots
    plt.savefig('psd-raw.png')

    # Display the plots
    plt.show()


# Function to plot the PSDs with markers
def plot_with_markers(welch_psd_vals, fft_psd_vals, freq_shift, amp_shift, phase_shift):

    # Plot the two PSDs on the same plot
    plt.plot(welch_psd_vals, 'b', label='Welch')
    plt.plot(fft_psd_vals, 'r', label='FFT')

    # Set the labels
    plt.ylabel('PSD')
    plt.xlabel('Frequency')

    # Add a marker line for the frequency shift
    plt.axvline(x=freq_shift, color='green',
                linestyle='--', label='Frequency Shift')

    # Add a marker line for the amplitude shift
    plt.axhline(y=amp_shift, color='purple',
                linestyle='--', label='Amplitude Shift')

    # Add a marker line for the phase shift
    plt.axhline(y=phase_shift, color='yellow',
                linestyle='--', label='Phase Shift')

    # Add a legend and set the location to the upper right
    # Include the marker lines in the legend
    plt.legend(loc='upper right', markerscale=0.5, handlelength=1.5)

    # Add a title to the plot
    plt.title('PSD Comparison with Shifts')

    # Add a grid to the plot
    plt.grid()

    # Tighten the layout
    plt.tight_layout()

    # Save the plots
    plt.savefig('psd-markers.png')

    # Display the plots
    plt.show()


# Function to display the differences between the two PSD methods
def display_psd_diffs():
    # Generate a signal
    signal, freq_shift, amp_shift, phase_shift = generate_signal()

    # Calculate the PSDs
    welch_psd_vals = welch_psd(signal, len(signal))
    fft_psd_vals = fft_psd(signal, len(signal))

    # Plot the original signals
    plot_signal(signal)

    # Plot the PSDs without markers
    plot_without_markers(welch_psd_vals, fft_psd_vals)

    # Plot the PSDs with markers
    plot_with_markers(welch_psd_vals, fft_psd_vals,
                      freq_shift, amp_shift, phase_shift)


# display_psd_diffs()

# import statements for matplotlib animations

# Function to plot a signal and draw lines between parts of some peaks in the signal

def signal_feature_animation(signal, filename, xlabel='Time', ylabel='Signal', title='Original Signal'):

    # First plot the signal
    plt.plot(signal, "purple")

    # Set the labels
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Time')

    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Signal')

    # Add a title to the plot, if one is provided
    if title:
        plt.title(title)
    else:
        plt.title('Original Signal')

    # Add a grid to the plot
    plt.grid()

    # Tighten the layout
    plt.tight_layout()

    # Set two dots to be the markers between which the lines will be drawn
    # These dots will be animated
    dot1, = plt.plot(0, 0, 'ko')

    # dot2 will be a few points ahead of dot1 in the signal
    dot2, = plt.plot(0, 0, 'ko')

    # Set the lines to be drawn between the two dots
    line, = plt.plot([0, 0], [0, 0], 'blue')

    # Animate the dots to move along the signal
    def animate(i):
        # Set the first dot to be the ith point in the signal
        dot1.set_data(i, signal[i])

        # Set the second dot to be the ith+10 point in the signal
        dot2.set_data(i+10, signal[i+10])

        # Set the line to be drawn between the two dots
        line.set_data([i, i+10], [signal[i], signal[i+10]])

        # Return the dots and line
        return dot1, dot2, line

    # Create the animation
    # slow down the animation by a factor of 10 to make it easier to see
    # fps is the number of frames per second, set to 5
    # blit=True means only re-draw the parts that have changed

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(signal)-10, interval=100, blit=True)

    # Save the animation
    anim.save(filename, writer='imagemagick')

    # Display the animation
    plt.show()


a_signal, freq_shift, amp_shift, phase_shift = generate_signal()

# Draw the animation for the plain signal
signal_feature_animation(a_signal, "signal_feature_animation.gif")

# make the fft psd
fft_psd_vals = fft_psd(a_signal, len(a_signal))

# Draw the animation for the fft psd
signal_feature_animation(
    fft_psd_vals, "fft_psd_animation.gif", xlabel='Frequency', ylabel='PSD', title='FFT PSD')

# make the welch psd
welch_psd_vals = welch_psd(a_signal, len(a_signal))

# Draw the animation for the welch psd
signal_feature_animation(
    welch_psd_vals, "welch_psd_animation.gif", xlabel='Frequency', ylabel='PSD', title='Welch PSD')
