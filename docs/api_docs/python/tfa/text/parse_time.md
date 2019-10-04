<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.parse_time" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.parse_time


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/text/parse_time_op.py#L30-L86">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Parse an input string according to the provided format string into a

### Aliases:

* `tfa.text.parse_time_op.parse_time`


``` python
tfa.text.parse_time(
    time_string,
    time_format,
    output_unit
)
```



<!-- Placeholder for "Used in" -->
Unix time.

Parse an input string according to the provided format string into a Unix
time, the number of seconds / milliseconds / microseconds / nanoseconds
elapsed since January 1, 1970 UTC.

Uses strftime()-like formatting options, with the same extensions as
FormatTime(), but with the exceptions that %E#S is interpreted as %E*S, and
%E#f as %E*f.  %Ez and %E*z also accept the same inputs.

%Y consumes as many numeric characters as it can, so the matching
data should always be terminated with a non-numeric. %E4Y always
consumes exactly four characters, including any sign.

Unspecified fields are taken from the default date and time of ...

  "1970-01-01 00:00:00.0 +0000"

For example, parsing a string of "15:45" (%H:%M) will return an
Unix time that represents "1970-01-01 15:45:00.0 +0000".

Note that ParseTime only heeds the fields year, month, day, hour,
minute, (fractional) second, and UTC offset.  Other fields, like
weekday (%a or %A), while parsed for syntactic validity, are
ignored in the conversion.

Date and time fields that are out-of-range will be treated as
errors rather than normalizing them like `absl::CivilSecond` does.
For example, it is an error to parse the date "Oct 32, 2013"
because 32 is out of range.

A leap second of ":60" is normalized to ":00" of the following
minute with fractional seconds discarded.  The following table
shows how the given seconds and subseconds will be parsed:

  "59.x" -> 59.x  // exact
  "60.x" -> 00.0  // normalized
  "00.x" -> 00.x  // exact

#### Args:


* <b>`time_string`</b>: The input time string to be parsed.
* <b>`time_format`</b>: The time format.
* <b>`output_unit`</b>: The output unit of the parsed unix time. Can only be SECOND,
  MILLISECOND, MICROSECOND, NANOSECOND.


#### Returns:

the number of seconds / milliseconds / microseconds / nanoseconds elapsed
  since January 1, 1970 UTC.



#### Raises:


* <b>`ValueError`</b>: If `output_unit` is not a valid value,
  if parsing `time_string` according to `time_format` failed.